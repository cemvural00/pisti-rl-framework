"""Play Pişti against an agent in the terminal.

Usage:
    venv/bin/python scripts/play.py                       # vs trained agent
    venv/bin/python scripts/play.py --agent expectimax    # vs search agent
    venv/bin/python scripts/play.py --agent greedy
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import CARD_POINTS, PistiGame, card_name, new_deck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="ppo:runs/ppo_main/final_model")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    from training.evaluate import build_agent

    name, agent, agent_obs = build_agent(args.agent, seed=args.seed or 0)
    rng = random.Random(args.seed)

    human_score_total, agent_score_total = 0, 0
    game_no = 0
    while True:
        game_no += 1
        human = game_no % 2  # alternate seats; human is this player id
        game = PistiGame(deck=new_deck(rng), first_player=(game_no + 1) % 2)
        if hasattr(agent, "reset"):
            agent.reset()
        print(
            f"\n=== Game {game_no} — you are Player {human}, "
            f"{'you lead' if game.current == human else f'{name} leads'} ==="
        )

        while not game.done:
            p_id = game.current
            if p_id == human:
                top = card_name(game.table[-1]) if game.table else "(empty)"
                hidden = f" +{len(game.hidden_center)} hidden" if game.hidden_center else ""
                print(f"\ntable [{len(game.table)} cards{hidden}], top: {top}")
                print(
                    f"score: you {game.points[human]} — {name} {game.points[1-human]}"
                    f"   captured: {game.captured_count[human]}-{game.captured_count[1-human]}"
                    f"   stock: {len(game.stock)}"
                )
                hand = sorted(game.hands[human])
                for i, c in enumerate(hand):
                    pts = f" ({CARD_POINTS[c]}p)" if CARD_POINTS[c] else ""
                    print(f"  [{i}] {card_name(c)}{pts}")
                while True:
                    try:
                        choice = input("play which card? ")
                        card = hand[int(choice)]
                        break
                    except (ValueError, IndexError):
                        print("enter a number from the list")
                    except (EOFError, KeyboardInterrupt):
                        print("\nbye!")
                        return
                info = game.step(card)
                if info["pisti"]:
                    print(f"  >>> PIŞTİ! +{10 * info['pisti']} points <<<")
                elif info["captured"]:
                    print(f"  you captured the pile (+{info['points_gained']} pts)")
            else:
                obs = agent_obs.encode(game, p_id)
                if getattr(agent, "wants_game", False):
                    a = agent.predict(obs, obs["action_mask"], game=game, player=p_id)
                else:
                    a = agent.predict(obs, obs["action_mask"])
                info = game.step(int(a))
                msg = f"{name} plays {card_name(int(a))}"
                if info["pisti"]:
                    msg += f"  >>> PIŞTİ for {name}! <<<"
                elif info["captured"]:
                    msg += "  (captures the pile)"
                print(msg)

        s = game.scores()
        you, them = s[human], s[1 - human]
        human_score_total += you
        agent_score_total += them
        verdict = "you win!" if you > them else ("tie" if you == them else f"{name} wins")
        print(f"\n=== final: you {you} — {name} {them}  ({verdict}) ===")
        print(f"running total: you {human_score_total} — {name} {agent_score_total}")
        try:
            if input("\nplay again? [y/n] ").strip().lower() not in ("y", "yes", ""):
                break
        except (EOFError, KeyboardInterrupt):
            break
    print("bye!")


if __name__ == "__main__":
    main()
