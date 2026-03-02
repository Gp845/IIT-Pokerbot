'''
Advanced Sneak Peek Hold'em bot with:
  - Counterfactual regret minimization (CFR) inspired strategy
  - Rich opponent modelling (VPIP, aggression factor, per-street fold-to-bet,
    showdown hand-strength history)
  - Monte Carlo equity estimation + Chen-formula pre-flop lookup
  - Optimal second-price (Vickrey) auction bidding
  - Position-aware, bankroll-aware, and time-bank-aware decision making
'''

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from treys import Card, Evaluator, Deck
from collections import defaultdict
import random

# ---------------------------------------------------------------------------
# Strategic constants
# ---------------------------------------------------------------------------

_PRIOR_FOLD_RATE = 0.35       # assumed fold-to-bet before we have enough data
_MIN_STREET_SAMPLES = 5       # minimum observed bets before using per-street fold rate
_PRIOR_BLUFF_RATE = 0.15      # assumed showdown bluff rate before we have enough data
_MIN_SHOWDOWN_SAMPLES = 5     # minimum showdowns before using measured bluff rate
_MAX_REGRET = 200.0           # regret clamp; prevents runaway divergence in online CFR
_MAX_STREET_RAISES = 3        # stop escalating past this many raises per street
_FORCE_PASSIVE_THRESHOLD = 0.82  # equity below which we stop raising after MAX_STREET_RAISES
_CLOSE_SPOT_THRESHOLD = 0.10  # ev_raise within this fraction of pot → use CFR to decide
_MC_SAMPLES_FLOP_TURN = 50    # Monte Carlo samples on flop / turn
_MC_SAMPLES_RIVER = 60        # Monte Carlo samples on river


# ---------------------------------------------------------------------------
# Pre-flop hand-strength helpers (Chen formula)
# ---------------------------------------------------------------------------

_RANK_VAL = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
}
_HIGH_SCORE = {14: 10, 13: 8, 12: 7, 11: 6, 10: 5, 9: 4.5, 8: 4,
               7: 3.5, 6: 3, 5: 2.5, 4: 2, 3: 1.5, 2: 1}


def _chen_score(c1: str, c2: str) -> float:
    """Return Chen formula score for two hole-card strings, e.g. 'Ah', 'Kd'."""
    r1, r2 = _RANK_VAL[c1[0]], _RANK_VAL[c2[0]]
    suited = c1[1] == c2[1]
    if r1 < r2:
        r1, r2 = r2, r1
    score = _HIGH_SCORE.get(r1, 0)
    if r1 == r2:
        score = max(score * 2, 5)
    else:
        if suited:
            score += 2
        gap = r1 - r2 - 1
        if gap == 1:
            score -= 1
        elif gap == 2:
            score -= 2
        elif gap == 3:
            score -= 4
        elif gap >= 4:
            score -= 5
        if gap <= 2 and r1 <= 12:
            score += 1
    return score


def preflop_strength(hand: list) -> float:
    """Normalise Chen score to [0, 1]; ~20 is the best possible (AA)."""
    if len(hand) < 2:
        return 0.5
    raw = _chen_score(hand[0], hand[1])
    return max(0.0, min(1.0, (raw + 1.0) / 21.0))


# ---------------------------------------------------------------------------
# Opponent model
# ---------------------------------------------------------------------------

class OpponentModel:
    """Tracks opponent statistics per-street and infers their tendencies."""

    def __init__(self):
        self.total_hands = 0
        self.vpip_hands = 0          # voluntarily put chips in pre-flop

        self.total_raises = 0
        self.total_calls = 0
        self.total_folds = 0

        # Per-street fold-to-aggression counts
        self.folds_to_bet: dict = defaultdict(int)
        self.bets_faced: dict = defaultdict(int)

        self._prev_opp_wager = 0
        self._prev_street = None

        # Showdown tracking: list of (opp_wager_at_showdown, opp_hand_rank_pct)
        # rank_pct 0=best (royal flush) → 1=worst (high card)
        self.showdown_samples: list = []

    # ── derived stats ──────────────────────────────────────────────────────

    @property
    def vpip(self) -> float:
        return self.vpip_hands / max(1, self.total_hands)

    @property
    def aggression_factor(self) -> float:
        """Raises / (Calls + Folds); >1 means aggressive."""
        denom = self.total_calls + self.total_folds
        return self.total_raises / max(1, denom)

    @property
    def overall_fold_rate(self) -> float:
        total = self.total_folds + self.total_calls + self.total_raises
        return self.total_folds / max(1, total)

    def fold_rate_street(self, street: str) -> float:
        faced = self.bets_faced.get(street, 0)
        folds = self.folds_to_bet.get(street, 0)
        if faced < _MIN_STREET_SAMPLES:
            return _PRIOR_FOLD_RATE
        return folds / faced

    def is_aggressive(self) -> bool:
        return self.aggression_factor > 1.5 and self.total_hands > 20

    def is_passive(self) -> bool:
        return self.aggression_factor < 0.8 and self.total_hands > 20

    # ── update ─────────────────────────────────────────────────────────────

    def observe(self, street: str, opp_wager: int, my_wager: int) -> None:
        """Call each time we observe an opponent wager increase."""
        if street != self._prev_street:
            self._prev_opp_wager = 0
            self._prev_street = street

        increase = opp_wager - self._prev_opp_wager
        if increase > 0:
            if opp_wager > my_wager:
                self.total_raises += 1
                self.bets_faced[street] += 1
            else:
                self.total_calls += 1
            if street == 'pre-flop' and increase > 0:
                self.vpip_hands += 1

        self._prev_opp_wager = opp_wager

    def observe_fold(self, street: str, my_wager: int) -> None:
        self.total_folds += 1
        if my_wager > 0:
            self.folds_to_bet[street] += 1

    def observe_showdown(self, opp_wager: int, opp_rank_pct: float) -> None:
        """Record a showdown: wager size vs hand-strength percentile (0=nuts, 1=worst)."""
        self.showdown_samples.append((opp_wager, opp_rank_pct))

    @property
    def showdown_bluff_rate(self) -> float:
        """Fraction of showdowns where opponent showed a weak hand (rank_pct > 0.7)."""
        if len(self.showdown_samples) < _MIN_SHOWDOWN_SAMPLES:
            return _PRIOR_BLUFF_RATE
        weak = sum(1 for _, r in self.showdown_samples if r > 0.7)
        return weak / len(self.showdown_samples)


# ---------------------------------------------------------------------------
# CFR-inspired regret tracking
# ---------------------------------------------------------------------------

class CFRTracker:
    """
    Lightweight online CFR over a compact abstract state space.

    State key: (street, equity_bucket[0-2], in_position, opp_type[0-2])
    Actions: 0=fold, 1=call/check, 2=raise

    After each hand we update regrets from the EV context stored per decision.
    Strategy = regret-matching (positive regrets normalised).
    """

    def __init__(self):
        self._regrets: dict = defaultdict(lambda: [0.0, 0.0, 0.0])
        self._strategy_sum: dict = defaultdict(lambda: [0.0, 0.0, 0.0])

    def state_key(self, street: str, equity: float,
                  in_position: bool, opp_type: int) -> tuple:
        eq_bucket = min(2, int(equity * 3))   # 0=weak, 1=mid, 2=strong
        return (street, eq_bucket, int(in_position), opp_type)

    def strategy(self, key: tuple) -> list:
        """Regret-matching: returns [p_fold, p_call, p_raise]."""
        pos = [max(0.0, r) for r in self._regrets[key]]
        total = sum(pos)
        if total > 0:
            return [p / total for p in pos]
        return [1 / 3, 1 / 3, 1 / 3]

    def sample_action(self, key: tuple,
                      can_fold: bool, can_call: bool, can_raise: bool) -> int:
        probs = self.strategy(key)
        if not can_fold:
            probs[0] = 0.0
        if not can_call:
            probs[1] = 0.0
        if not can_raise:
            probs[2] = 0.0
        total = sum(probs)
        if total == 0:
            return 1 if can_call else 0
        probs = [p / total for p in probs]
        r = random.random()
        cumul = 0.0
        for i, p in enumerate(probs):
            cumul += p
            if r <= cumul:
                return i
        return 1

    def update(self, key: tuple, action_taken: int,
               ev_fold: float, ev_call: float, ev_raise: float) -> None:
        """Update regrets; called at end of hand for each recorded decision.

        Regret for action i = ev[i] - ev[action_taken].
        Positive regret means action i would have been better than what we did,
        so regret-matching will increase its probability.
        """
        evs = [ev_fold, ev_call, ev_raise]
        ev_taken = evs[action_taken]
        regrets = self._regrets[key]
        for i in range(3):
            regrets[i] += evs[i] - ev_taken
            # Clamp to prevent divergence (see _MAX_REGRET)
            regrets[i] = max(-_MAX_REGRET, min(_MAX_REGRET, regrets[i]))
        strat = self.strategy(key)
        ssum = self._strategy_sum[key]
        for i in range(3):
            ssum[i] += strat[i]


# ---------------------------------------------------------------------------
# Main player class
# ---------------------------------------------------------------------------

class Player(BaseBot):
    '''
    A pokerbot.
    '''

    def __init__(self) -> None:
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.evaluator = Evaluator()
        self.opp_model = OpponentModel()
        self.cfr = CFRTracker()

        # Per-hand state
        self.round_num = 0
        self.bankroll = 0
        self.is_bb = False
        self.my_street_raises = 0
        self._last_street = None
        self._last_opp_wager = 0
        self._last_my_wager = 0

        # Decisions recorded this hand for CFR update in on_hand_end
        # Each entry: (cfr_key, action_index, ev_fold, ev_call, ev_raise)
        self._decisions: list = []
        self._time_bank: float = 20.0   # updated each action from game_info

    # -----------------------------------------------------------------------
    # Hand lifecycle
    # -----------------------------------------------------------------------

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_info: the GameInfo object.
        current_state: the PokerState object.

        Returns:
        Nothing.
        '''
        self.round_num = game_info.round_num
        self.bankroll = game_info.bankroll
        self.is_bb = current_state.is_bb
        self.my_street_raises = 0
        self._last_street = None
        self._last_opp_wager = 0
        self._last_my_wager = 0
        self._decisions = []
        self.opp_model.total_hands += 1

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_info: the GameInfo object.
        current_state: the PokerState object.

        Returns:
        Nothing.
        '''
        opp_cards = current_state.opp_revealed_cards
        opp_wager = current_state.opp_wager     # chips opponent put in this street
        street = current_state.street

        # ── Detect opponent fold (won without showdown) ───────────────────────
        if current_state.payoff > 0 and len(opp_cards) < 2:
            self.opp_model.observe_fold(street, self._last_my_wager)

        # ── Showdown analysis: learn hand strength opponent showed up with ─────
        # At showdown both hole cards are revealed (len == 2) and the board has 5 cards.
        if len(opp_cards) == 2 and len(current_state.board) == 5:
            try:
                opp_treys = [Card.new(c) for c in opp_cards]
                board_treys = [Card.new(c) for c in current_state.board]
                rank = self.evaluator.evaluate(board_treys, opp_treys)
                # treys rank: 1=best, 7462=worst → normalise to [0, 1] (0=nuts)
                rank_pct = rank / 7462.0
                self.opp_model.observe_showdown(opp_wager, rank_pct)
            except Exception:
                pass

        # ── Update CFR regrets for every decision made this hand ─────────────
        for key, action, ev_f, ev_c, ev_r in self._decisions:
            self.cfr.update(key, action, ev_f, ev_c, ev_r)

        # ── Reset per-hand tracking ───────────────────────────────────────────
        self._last_opp_wager = 0
        self._last_my_wager = 0
        self._last_street = None

    # -----------------------------------------------------------------------
    # Equity estimation
    # -----------------------------------------------------------------------

    def _mc_samples(self, base: int = 60) -> int:
        """Return number of Monte Carlo samples scaled to remaining time budget.

        The 20 s total budget across 1 000 rounds is ~20 ms per round on average.
        These thresholds are absolute remaining-time checks (in seconds): we scale
        down aggressively once fewer than 5 s remain to avoid time forfeit.
        """
        if self._time_bank > 10.0:
            return base
        if self._time_bank > 5.0:
            return max(20, base // 2)
        return max(10, base // 4)

    def _monte_carlo_equity(self, my_hand: list, board: list,
                            opp_hand: list = None, samples: int = 60) -> float:
        """Run Monte Carlo to estimate win probability."""
        if not my_hand or len(my_hand) < 2:
            return 0.5
        try:
            p1 = [Card.new(c) for c in my_hand]
            board_cards = [Card.new(c) for c in board]
            p2_known = [Card.new(c) for c in opp_hand] if opp_hand else None
        except Exception:
            return 0.5

        full_deck = Deck().cards
        used = set(p1 + board_cards)
        if p2_known:
            used.update(p2_known)
        remaining = [c for c in full_deck if c not in used]

        needed_board = 5 - len(board_cards)
        if p2_known and len(p2_known) == 2:
            n_draw = needed_board
        elif p2_known and len(p2_known) == 1:
            n_draw = 1 + needed_board
        else:
            n_draw = 2 + needed_board

        if len(remaining) < n_draw:
            return 0.5

        wins = 0.0
        for _ in range(samples):
            sample = random.sample(remaining, n_draw)
            if p2_known and len(p2_known) == 2:
                opp = p2_known
                sim_board = board_cards + sample[:needed_board]
            elif p2_known and len(p2_known) == 1:
                opp = p2_known + [sample[0]]
                sim_board = board_cards + sample[1:1 + needed_board]
            else:
                opp = sample[:2]
                sim_board = board_cards + sample[2:2 + needed_board]
            s1 = self.evaluator.evaluate(sim_board, p1)
            s2 = self.evaluator.evaluate(sim_board, opp)
            if s1 < s2:
                wins += 1.0
            elif s1 == s2:
                wins += 0.5
        return wins / samples

    def _get_equity(self, current_state: PokerState) -> float:
        """Return equity, using Chen lookup pre-flop and Monte Carlo post-flop."""
        street = current_state.street
        hand = current_state.my_hand
        board = current_state.board
        opp_cards = current_state.opp_revealed_cards

        if street == 'pre-flop' and not board:
            return preflop_strength(hand)

        base = _MC_SAMPLES_FLOP_TURN if street in ('flop', 'turn') else _MC_SAMPLES_RIVER
        return self._monte_carlo_equity(hand, board, opp_cards, self._mc_samples(base))

    # -----------------------------------------------------------------------
    # Auction strategy
    # -----------------------------------------------------------------------

    def _auction_bid(self, current_state: PokerState) -> int:
        """
        Bid our true value of information — dominant strategy in a second-price auction.

        True value ≈ information gain in chips = pot × info_sensitivity × 0.15
        Information sensitivity peaks at equity ≈ 0.5 (max decision uncertainty).
        """
        eq = self._monte_carlo_equity(
            current_state.my_hand, current_state.board,
            samples=self._mc_samples(_MC_SAMPLES_FLOP_TURN)
        )
        # Peaks at equity=0.5, falls to 0 at 0 or 1
        info_sensitivity = 1.0 - abs(eq - 0.5) * 2.0
        pot = current_state.pot
        chips = current_state.my_chips

        true_value = int(pot * 0.15 * info_sensitivity)
        bid = min(true_value, pot, chips)          # info cannot exceed the pot
        jitter = random.randint(-1, 2)              # prevent deterministic pattern
        return max(0, bid + jitter)

    # -----------------------------------------------------------------------
    # Main decision function
    # -----------------------------------------------------------------------

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_info: the GameInfo object.
        current_state: the PokerState object.

        Returns:
        Your action.
        '''
        street = current_state.street

        # ── REFRESH TIME BUDGET ───────────────────────────────────────────────
        self._time_bank = game_info.time_bank

        # ── AUCTION ──────────────────────────────────────────────────────────
        if street == 'auction':
            return ActionBid(self._auction_bid(current_state))

        # ── PER-STREET RESET ─────────────────────────────────────────────────
        if street != self._last_street:
            self._last_street = street
            self._last_opp_wager = 0
            self._last_my_wager = 0
            self.my_street_raises = 0

        # ── OPPONENT TRACKING ────────────────────────────────────────────────
        opp_wager = current_state.opp_wager
        my_wager = current_state.my_wager
        self.opp_model.observe(street, opp_wager, my_wager)
        self._last_opp_wager = opp_wager
        self._last_my_wager = my_wager

        # ── EQUITY ───────────────────────────────────────────────────────────
        equity = self._get_equity(current_state)

        # ── AVAILABLE ACTIONS ────────────────────────────────────────────────
        can_fold = current_state.can_act(ActionFold)
        can_check = current_state.can_act(ActionCheck)
        can_call_action = current_state.can_act(ActionCall)
        can_call = can_call_action or can_check
        can_raise = current_state.can_act(ActionRaise)
        min_raise, max_raise = current_state.raise_bounds if can_raise else (0, 0)

        pot = current_state.pot
        call_cost = current_state.cost_to_call

        # ── OPPONENT PROFILE ─────────────────────────────────────────────────
        if self.opp_model.is_aggressive():
            opp_type = 2
        elif self.opp_model.is_passive():
            opp_type = 0
        else:
            opp_type = 1

        # Showdown bluff rate: if opponent bluffs often at showdown,
        # we should call them down more and bluff them less.
        opp_bluff_rate = self.opp_model.showdown_bluff_rate

        # In Sneak Peek Hold'em the BB acts last post-flop → IP advantage
        in_position = self.is_bb
        cfr_key = self.cfr.state_key(street, equity, in_position, opp_type)

        # ── POT ODDS / FOLD LOGIC ─────────────────────────────────────────────
        if call_cost > 0 and can_fold:
            # Minimum equity needed to break even on a call
            pot_odds_eq = call_cost / (pot + call_cost)
            # Street-specific margin above pot odds
            margin = {'pre-flop': 0.06, 'flop': 0.07,
                      'turn': 0.08, 'river': 0.10}.get(street, 0.07)
            # Call wider against known bluffers (reduce margin proportionally)
            margin -= opp_bluff_rate * 0.05
            fold_threshold = pot_odds_eq + margin

            if equity < fold_threshold:
                shortfall = fold_threshold - equity
                fold_prob = min(0.92, shortfall * 6.0)
                if random.random() < fold_prob:
                    self._decisions.append((cfr_key, 0, 0.0, 0.0, -1.0))
                    return ActionFold()

        # ── EV CALCULATIONS ──────────────────────────────────────────────────
        ev_fold = 0.0
        # EV of call: equity × pot − (1−equity) × call_cost
        ev_call = equity * pot - (1.0 - equity) * call_cost if can_call else -1e9

        # EV of raise
        ev_raise = -1e9
        raise_size = 0
        force_passive = self.my_street_raises >= _MAX_STREET_RAISES and equity < _FORCE_PASSIVE_THRESHOLD

        if can_raise and not force_passive:
            # Bet sizing: value=0.75 pot, bluff/semi-bluff=0.65 pot, overbet nut river
            if equity > 0.80 and street == 'river':
                raise_mult = random.uniform(1.0, 1.3)
            elif equity > 0.65:
                raise_mult = 0.75
            elif equity > 0.45:
                raise_mult = 0.65   # semi-bluff / thin value
            else:
                raise_mult = 0.60   # bluff
            raise_size = int(pot * raise_mult)
            raise_size = max(min_raise, min(max_raise, raise_size))

            # Fold equity decreases with each additional raise this street
            base_fold = self.opp_model.fold_rate_street(street)
            fold_eq = base_fold * (0.65 ** self.my_street_raises)

            # EV(raise) = fold_eq×pot + (1−fold_eq)×EV(call after raise)
            ev_called = equity * (pot + raise_size * 2) - raise_size
            ev_raise = fold_eq * pot + (1.0 - fold_eq) * ev_called

        # ── CFR DECISION (use regret-matching in close spots) ────────────────
        # "Close" = ev_raise within _CLOSE_SPOT_THRESHOLD fraction of pot of ev_call
        close_spot = can_raise and not force_passive and abs(ev_raise - ev_call) < _CLOSE_SPOT_THRESHOLD * max(1, pot)
        if close_spot:
            cfr_action = self.cfr.sample_action(cfr_key, can_fold, can_call, can_raise)
        else:
            # EV-dominant choice
            best = max(ev_fold, ev_call, ev_raise)
            if best == ev_raise and can_raise and not force_passive:
                cfr_action = 2
            elif best == ev_call and can_call:
                cfr_action = 1
            elif can_fold:
                cfr_action = 0
            else:
                cfr_action = 1

        # Record decision for CFR update at hand end
        self._decisions.append((cfr_key, cfr_action, ev_fold, ev_call, ev_raise))

        # ── EXECUTE ACTION ───────────────────────────────────────────────────
        if cfr_action == 2 and can_raise and not force_passive:
            self.my_street_raises += 1
            bet = raise_size if raise_size > 0 else max(min_raise, int(pot * 0.65))
            return ActionRaise(max(min_raise, min(max_raise, bet)))

        if cfr_action == 0 and can_fold and call_cost > 0:
            return ActionFold()

        # Default: call / check
        if can_check:
            return ActionCheck()
        if can_call_action:
            return ActionCall()
        return ActionFold() if can_fold else ActionCall()


if __name__ == '__main__':
    run_bot(Player(), parse_args())