from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from treys import Card, Evaluator, Deck
import random
import numpy as np


class Player(BaseBot):

    def __init__(self):
        self.evaluator = Evaluator()
        self.my_bankroll = 0
        self.round_num = 0
        self.my_street_raises = 0

        # Opponent tendency tracking
        self.opp_folds = 0
        self.opp_calls = 0
        self.opp_raises = 0

        self.last_opp_wager = 0
        self.last_my_wager = 0
        self.last_street = ""


    # -------------------------------------------------
    # HAND LIFECYCLE
    # -------------------------------------------------

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState):
        self.my_bankroll = game_info.bankroll
        self.round_num = game_info.round_num

    def on_hand_end(self, game_info, current_state):
        # The only reliable place to catch a fold
        if current_state.payoff > 0 and not current_state.opp_revealed_cards:
            self.opp_folds += 1
        
        # Reset tracking variables for the next hand
        self.last_opp_wager = 0
        self.last_my_wager = 0
        self.last_street = ""

    # -------------------------------------------------
    # MONTE CARLO EQUITY
    # -------------------------------------------------

    def monte_carlo_equity(self, my_hand, board, opp_hand=None, samples=80):

        if not my_hand or len(my_hand) < 2:
            return 0.5

        p1 = [Card.new(c) for c in my_hand]
        board_cards = [Card.new(c) for c in board]

        p2_known = [Card.new(c) for c in opp_hand] if opp_hand else None

        full_deck = Deck().cards
        used = set(p1 + board_cards)
        if p2_known:
            used.update(p2_known)

        remaining = [c for c in full_deck if c not in used]

        wins = 0
        needed = 5 - len(board_cards)

        for _ in range(samples):
            sample = random.sample(remaining, len(remaining))

            if p2_known:
                opp = p2_known
                sim_board = board_cards + sample[:needed]
            else:
                opp = sample[:2]
                sim_board = board_cards + sample[2:2 + needed]

            s1 = self.evaluator.evaluate(sim_board, p1)
            s2 = self.evaluator.evaluate(sim_board, opp)

            if s1 < s2:
                wins += 1
            elif s1 == s2:
                wins += 0.5

        return wins / samples



    # -------------------------------------------------
    # AUCTION EXPLOIT STRATEGY
    # -------------------------------------------------

    def auction_strategy(self, my_hand, chips):

        ranks = [c[0] for c in my_hand]
        suits = [c[1] for c in my_hand]

        is_pair = ranks[0] == ranks[1]
        suited = suits[0] == suits[1]

        # Premium pairs slowplay (they play well anyway)
        if is_pair and ranks[0] in "AK":
            base = 0.03

        # Information-sensitive hands
        elif suited or is_pair:
            base = 0.12

        # Broadway
        elif all(r in "AKQJT" for r in ranks):
            base = 0.10

        # Single high card
        elif any(r in "AKQJT" for r in ranks):
            base = 0.01

        else:
            base = 0.00

        # Calculate bid WITH jitter first
        jitter = random.randint(-3, 3)
        bid = int(chips * base) + jitter
        
        # Apply the Pot Cap to the final bid
        # This prevents bidding 500 to win a 40 chip pot
        
        return max(0, bid)
    # -------------------------------------------------
    # MAIN DECISION LOGIC
    # -------------------------------------------------

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        street = current_state.street

        # ---------- AUCTION ----------
        if street == "auction":
            bid = self.auction_strategy(current_state.my_hand, current_state.my_chips)
            pot_cap = int(current_state.pot * 1.5) 
            bid = min(bid, pot_cap)
            return ActionBid(bid)

        # ---------- TRACKING & STATE MANAGEMENT ----------
        if street != self.last_street:
            self.last_opp_wager = 0
            self.last_my_wager = 0
            self.last_street = street
            self.my_street_raises = 0 

        if current_state.opp_wager > self.last_opp_wager:
            if current_state.opp_wager > self.last_my_wager:
                self.opp_raises += 1
            else:
                self.opp_calls += 1

        self.last_opp_wager = current_state.opp_wager
        self.last_my_wager = current_state.my_wager

        # ---------- EQUITY CALCULATIONS ----------
        opp_hand = current_state.opp_revealed_cards
        win_prob = self.monte_carlo_equity(current_state.my_hand, current_state.board, opp_hand, samples=100)

        if current_state.cost_to_call > 0:
            win_prob -= 0.12 
        if not opp_hand:
            win_prob -= 0.07

        win_prob = max(0, min(1, win_prob))

        # ---------- ACTION CONSTANTS ----------
        pot = current_state.pot
        call_cost = current_state.cost_to_call
        can_fold = current_state.can_act(ActionFold)
        can_call = current_state.can_act(ActionCall) or current_state.can_act(ActionCheck)
        can_raise = current_state.can_act(ActionRaise)
        min_raise, max_raise = current_state.raise_bounds if can_raise else (0, 0)

        # 1. STOP THE LOOP: Cap aggression unless we have a monster
        force_passive = False
        if self.my_street_raises >= 2 and win_prob < 0.85:
            force_passive = True
    
        # ---------- FOLD LOGIC (Pot Odds Aware) ----------
        if call_cost > 0 and can_fold:
            # 1. Determine the baseline threshold for this street
            thresh = {"pre-flop": 0.28, "flop": 0.30, "turn": 0.35, "river": 0.40}.get(street, 0.30)
            
            # 2. Calculate "Pot Odds" (the equity needed to break even on a call)
            # Example: call 10 into a 90 pot means you need 10/100 = 10% equity
            needed_equity = call_cost / (pot + call_cost) if (pot + call_cost) > 0 else 0
            
            # 3. Dynamic Threshold Adjustment: 
            # If the call is very cheap, lower the threshold so we don't fold too easily.
            # We take the lower of our "Strategy Threshold" and a padded version of "Pot Odds"
            dynamic_thresh = max(needed_equity + 0.05, thresh - 0.10) 
            
            # Final threshold is whichever is lower: our base strategy or the pot odds logic
            final_thresh = min(thresh, dynamic_thresh)

            if win_prob < final_thresh:
                # Use distance below the NEW final_thresh to determine fold probability
                diff = final_thresh - win_prob
                fold_probability = min(0.85, diff * 4.0)
                
                if random.random() < fold_probability:
                    return ActionFold()

        # ---------- STRATEGIC SPECIAL MOVES ----------
        
        # 2. FLOP SEMI-BLUFF & PURE BLUFF
        if street == "flop" and can_raise and not force_passive:
            total_act = self.opp_folds + self.opp_calls + self.opp_raises
            fold_rate = (self.opp_folds / total_act) if total_act > 20 else 0.35
            
            # Semi-bluff (Draws) or Pure Bluff (Air vs Tight players)
            is_semi_bluff = (0.35 <= win_prob <= 0.55 and random.random() < 0.25)
            is_pure_bluff = (win_prob < 0.25 and fold_rate > 0.45 and random.random() < 0.15)
            
            if is_semi_bluff or is_pure_bluff:
                self.my_street_raises += 1
                bet = int(pot * random.uniform(0.6, 0.8))
                return ActionRaise(max(min_raise, min(max_raise, bet)))

        # 3. RIVER OVERBET LOGIC (Value & Polar Bluffs)
        if street == "river" and can_raise and not force_passive:
            # Value Overbet (The Nuts)
            if win_prob > 0.85:
                self.my_street_raises += 1
                bet = int(pot * random.uniform(1.1, 1.4))
                return ActionRaise(max(min_raise, min(max_raise, bet)))
            
            # Polar Bluff (Opponent folds too much and we have air)
            total_act = self.opp_folds + self.opp_calls + self.opp_raises
            fold_rate = (self.opp_folds / total_act) if total_act > 20 else 0.35
            if win_prob < 0.30 and fold_rate > 0.50 and random.random() < 0.15:
                self.my_street_raises += 1
                bet = int(pot * 1.2)
                return ActionRaise(max(min_raise, min(max_raise, bet)))

        # ---------- EV CALCULATIONS ----------
        ev_fold = 0
        ev_call = win_prob * pot - (1 - win_prob) * call_cost if can_call else -float('inf')
        ev_raise = -float("inf")

        if can_raise and not force_passive:
            total_actions = self.opp_folds + self.opp_calls + self.opp_raises
            base_fold = (self.opp_folds / total_actions) if total_actions > 20 else 0.35
            fold_prob = base_fold * (0.4 ** self.my_street_raises)
            
            raise_size = int(pot * 0.75)
            raise_size = max(min_raise, min(max_raise, raise_size))
            ev_showdown = (win_prob * (pot + raise_size)) - ((1 - win_prob) * raise_size)
            ev_raise = (fold_prob * pot) + (1 - fold_prob) * ev_showdown

        # ---------- FINAL DECISION ----------
        best_ev = max(ev_fold, ev_call, ev_raise)

        if best_ev == ev_raise and can_raise and not force_passive:
            self.my_street_raises += 1
            bet = int(pot * 0.6)
            return ActionRaise(max(min_raise, min(max_raise, bet)))

        if (best_ev == ev_call or force_passive) and can_call:
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionCall()

        return ActionFold() if can_fold else ActionCall()


if __name__ == "__main__":
    run_bot(Player(), parse_args())