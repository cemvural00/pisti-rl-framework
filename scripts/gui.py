"""Browser GUI for playing Pişti against any agent.

    venv/bin/python scripts/gui.py            # then open http://localhost:8777
    venv/bin/python scripts/gui.py --port 9000
"""

import argparse
import glob
import os
import random
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request

from encoding.obs import Observer
from engine.game import CARD_POINTS, PistiGame, card_name, new_deck, rank_of
from training.evaluate import build_agent

app = Flask(__name__)


def available_agents():
    agents = [
        ("greedy", "Greedy (heuristic)"),
        ("hunter", "Pişti Hunter (heuristic)"),
        ("expectimax:16,6", "Expectimax (search)"),
        ("random", "Random"),
    ]
    for path in sorted(glob.glob("runs/*/final_model.zip")):
        run = os.path.basename(os.path.dirname(path))
        spec = f"dqn:{path[:-4]}" if "dqn" in run else f"ppo:{path[:-4]}"
        if "nomem" in run:
            spec = f"ppo-nomem:{path[:-4]}"
        agents.append((spec, f"{run} (trained RL)"))
    return agents


class Session:
    """One human-vs-agent session (single user, in-memory)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rng = random.Random()
        self.agent = None
        self.agent_obs = Observer()
        self.agent_name = None
        self.game = None
        self.human = 0
        self.game_no = 0
        self.totals = [0, 0]  # [human, agent]
        self.log = []

    def new_game(self, spec=None):
        if spec is not None:
            self.agent_name, self.agent, self.agent_obs = build_agent(spec, seed=0)
            self.game_no = 0
            self.totals = [0, 0]
        self.game_no += 1
        self.human = self.game_no % 2
        self.game = PistiGame(
            deck=new_deck(self.rng), first_player=(self.game_no + 1) % 2
        )
        if hasattr(self.agent, "reset"):
            self.agent.reset()
        self.log = [f"Game {self.game_no} — {'you lead' if self.game.current == self.human else self.agent_name + ' leads'}"]
        events = []
        if self.game.current != self.human:
            events += self._agent_moves()
        return events

    def _agent_moves(self):
        events = []
        g = self.game
        while not g.done and g.current != self.human:
            p = g.current
            obs = self.agent_obs.encode(g, p)
            if getattr(self.agent, "wants_game", False):
                a = self.agent.predict(obs, obs["action_mask"], game=g, player=p)
            else:
                a = self.agent.predict(obs, obs["action_mask"])
            info = g.step(int(a))
            events.append(self._event(p, int(a), info))
        return events

    def play(self, card):
        g = self.game
        if g is None or g.done or g.current != self.human:
            raise ValueError("not your turn")
        if card not in g.hands[self.human]:
            raise ValueError("card not in hand")
        info = g.step(card)
        events = [self._event(self.human, card, info)]
        events += self._agent_moves()
        if g.done:
            s = g.scores()
            self.totals[0] += s[self.human]
            self.totals[1] += s[1 - self.human]
        return events

    def _event(self, player, card, info):
        who = "you" if player == self.human else self.agent_name
        e = {
            "who": who,
            "human": player == self.human,
            "card": card,
            "card_name": card_name(card),
            "captured": info["captured"],
            "pisti": info["pisti"],
            "points": info["points_gained"],
        }
        msg = f"{who}: {card_name(card)}"
        if info["pisti"] == 2:
            msg += "  ‼️ DOUBLE PIŞTİ +20"
        elif info["pisti"] == 1:
            msg += "  ⚡ PIŞTİ +10"
        elif info["captured"]:
            msg += f"  (captured, +{info['points_gained']})"
        self.log.append(msg)
        return e

    def state(self):
        g = self.game
        h, a = self.human, 1 - self.human
        s = g.scores()
        winner = None
        if g.done:
            winner = "you" if g.winner() == h else ("tie" if g.winner() is None else self.agent_name)
        return {
            "agent": self.agent_name,
            "game_no": self.game_no,
            "hand": sorted(g.hands[h]),
            "hand_names": {c: card_name(c) for c in g.hands[h]},
            "card_points": {c: CARD_POINTS[c] for c in g.hands[h]},
            "opp_hand_count": len(g.hands[a]),
            "table_top": g.table[-1] if g.table else None,
            "table_top_name": card_name(g.table[-1]) if g.table else None,
            "table_count": len(g.table),
            "hidden_count": len(g.hidden_center),
            "stock": len(g.stock),
            "your_points": g.points[h],
            "agent_points": g.points[a],
            "your_captured": g.captured_count[h],
            "agent_captured": g.captured_count[a],
            "your_pistis": g.pistis[h] + g.double_pistis[h],
            "agent_pistis": g.pistis[a] + g.double_pistis[a],
            "your_turn": (not g.done) and g.current == h,
            "done": g.done,
            "final_scores": [s[h], s[a]] if g.done else None,
            "winner": winner,
            "totals": self.totals,
            "log": self.log[-14:],
        }


SESSION = Session()


@app.route("/")
def index():
    return PAGE


@app.route("/api/agents")
def api_agents():
    return jsonify(available_agents())


@app.route("/api/new", methods=["POST"])
def api_new():
    spec = request.json.get("agent")
    with SESSION.lock:
        events = SESSION.new_game(spec)
        return jsonify({"events": events, "state": SESSION.state()})


@app.route("/api/next", methods=["POST"])
def api_next():
    with SESSION.lock:
        events = SESSION.new_game(None)
        return jsonify({"events": events, "state": SESSION.state()})


@app.route("/api/play", methods=["POST"])
def api_play():
    card = int(request.json["card"])
    with SESSION.lock:
        try:
            events = SESSION.play(card)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"events": events, "state": SESSION.state()})


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pişti RL</title>
<style>
  :root { --felt:#1b6b45; --felt-dark:#14543a; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,'Segoe UI',sans-serif;
         background:radial-gradient(ellipse at center, var(--felt) 0%, var(--felt-dark) 100%);
         color:#fff; min-height:100vh; }
  #wrap { max-width:980px; margin:0 auto; padding:18px; }
  h1 { font-size:20px; margin:0 0 12px; font-weight:600; letter-spacing:.5px; }
  .bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
  select,button { font-size:15px; padding:8px 14px; border-radius:8px; border:none; cursor:pointer; }
  button { background:#ffd54f; color:#333; font-weight:600; }
  button:hover { background:#ffe082; }
  .scores { display:flex; gap:18px; margin:6px 0 12px; font-size:15px; flex-wrap:wrap; }
  .scores b { color:#ffd54f; }
  .zone { display:flex; align-items:center; gap:14px; min-height:130px; }
  .card { width:74px; height:108px; border-radius:9px; background:#fff; color:#111;
          display:inline-flex; flex-direction:column; justify-content:space-between;
          padding:7px 8px; font-size:21px; font-weight:700; box-shadow:0 3px 8px rgba(0,0,0,.45);
          position:relative; user-select:none; }
  .card.red { color:#c62828; }
  .card .pts { position:absolute; bottom:6px; right:8px; font-size:11px; color:#888; font-weight:600; }
  .card.back { background:repeating-linear-gradient(45deg,#27408b,#27408b 6px,#1d3370 6px,#1d3370 12px);
               border:3px solid #fff; }
  .card.empty { background:rgba(255,255,255,.12); border:2px dashed rgba(255,255,255,.4); box-shadow:none; }
  #hand .card { cursor:pointer; transition:transform .12s; }
  #hand .card:hover { transform:translateY(-12px); }
  #hand.locked .card { cursor:not-allowed; opacity:.75; }
  #hand.locked .card:hover { transform:none; }
  .pile { position:relative; }
  .pile .badge { position:absolute; top:-10px; right:-12px; background:#ffd54f; color:#333;
                 border-radius:11px; padding:2px 9px; font-size:13px; font-weight:700; }
  .hint { font-size:13px; opacity:.85; }
  #log { background:rgba(0,0,0,.30); border-radius:10px; padding:10px 14px; font-size:14px;
         min-height:90px; max-height:200px; overflow-y:auto; line-height:1.55; }
  #banner { position:fixed; top:34%; left:50%; transform:translate(-50%,-50%) scale(.6);
            font-size:58px; font-weight:800; color:#ffd54f; text-shadow:0 0 30px #000;
            opacity:0; pointer-events:none; transition:all .35s; z-index:5; }
  #banner.show { opacity:1; transform:translate(-50%,-50%) scale(1); }
  .row-label { width:90px; font-size:13px; opacity:.8; }
  #result { font-size:17px; font-weight:700; color:#ffd54f; margin:8px 0; }
</style></head>
<body><div id="wrap">
  <h1>♠ Pişti — you vs the machines</h1>
  <div class="bar">
    <select id="agent"></select>
    <button onclick="newMatch()">new match</button>
    <button id="nextBtn" onclick="nextGame()" style="display:none">next game ▶</button>
    <span class="hint" id="turnHint"></span>
  </div>
  <div class="scores" id="scores"></div>
  <div class="zone"><div class="row-label">opponent</div><div id="opp"></div></div>
  <div class="zone"><div class="row-label">table</div><div id="table"></div></div>
  <div class="zone"><div class="row-label">your hand</div><div id="hand"></div></div>
  <div id="result"></div>
  <div id="log"></div>
</div>
<div id="banner"></div>
<script>
const RED = c => { const s = Math.floor(c/13); return s===1||s===2; };
let busy = false;

async function api(path, body){ const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); return r.json(); }

function cardHTML(c, name, pts, onclick){
  const cls = 'card'+(RED(c)?' red':'');
  const suit = name.slice(-1), rank = name.slice(0,-1);
  return `<div class="${cls}" ${onclick?`onclick="${onclick}"`:''}>
    <div>${rank}<br>${suit}</div>${pts?`<span class="pts">+${pts}</span>`:''}</div>`;
}

function banner(text){
  const b = document.getElementById('banner');
  b.textContent = text; b.classList.add('show');
  setTimeout(()=>b.classList.remove('show'), 1400);
}

function render(st){
  document.getElementById('scores').innerHTML =
    `<span>game <b>${st.game_no}</b> vs <b>${st.agent}</b></span>`+
    `<span>points <b>${st.your_points}</b> : ${st.agent_points}</span>`+
    `<span>captured <b>${st.your_captured}</b> : ${st.agent_captured}</span>`+
    `<span>pişti <b>${st.your_pistis}</b> : ${st.agent_pistis}</span>`+
    `<span>stock ${st.stock}</span>`+
    `<span>match total <b>${st.totals[0]}</b> : ${st.totals[1]}</span>`;
  document.getElementById('opp').innerHTML =
    Array(st.opp_hand_count).fill('<div class="card back"></div>').join(' ') || '<span class="hint">no cards</span>';
  let t = '';
  if (st.table_top !== null){
    t = `<div class="pile">${cardHTML(st.table_top, st.table_top_name)}
         <span class="badge">${st.table_count}${st.hidden_count?'+'+st.hidden_count+'🂠':''}</span></div>`;
  } else t = '<div class="card empty"></div>';
  document.getElementById('table').innerHTML = t;
  const hand = document.getElementById('hand');
  hand.innerHTML = st.hand.map(c=>cardHTML(c, st.hand_names[c], st.card_points[c], `play(${c})`)).join(' ')
                   || '<span class="hint">no cards</span>';
  hand.className = st.your_turn && !st.done ? '' : 'locked';
  document.getElementById('turnHint').textContent =
    st.done ? '' : (st.your_turn ? 'your move — click a card' : 'opponent thinking…');
  document.getElementById('log').innerHTML = st.log.map(l=>'<div>'+l+'</div>').join('');
  document.getElementById('log').scrollTop = 1e9;
  const res = document.getElementById('result');
  if (st.done){
    res.textContent = `final ${st.final_scores[0]} : ${st.final_scores[1]} — ` +
      (st.winner==='you' ? 'you win! 🎉' : st.winner==='tie' ? 'tie' : st.winner+' wins 🤖');
    document.getElementById('nextBtn').style.display = '';
  } else { res.textContent=''; document.getElementById('nextBtn').style.display='none'; }
}

async function animate(events, st){
  for (const e of events){
    if (e.pisti===2) banner('DOUBLE PIŞTİ ‼️');
    else if (e.pisti===1) banner(e.human ? 'PIŞTİ! ⚡' : 'pişti for '+e.who+' 😬');
  }
  render(st);
}

async function newMatch(){
  const spec = document.getElementById('agent').value;
  const r = await api('/api/new', {agent: spec});
  await animate(r.events, r.state);
}
async function nextGame(){ const r = await api('/api/next'); await animate(r.events, r.state); }
async function play(c){
  if (busy) return; busy = true;
  try { const r = await api('/api/play', {card:c}); if (!r.error) await animate(r.events, r.state); }
  finally { busy = false; }
}
(async ()=>{
  const agents = await (await fetch('/api/agents')).json();
  document.getElementById('agent').innerHTML =
    agents.map(a=>`<option value="${a[0]}">${a[1]}</option>`).join('');
  const pref = agents.find(a=>a[0].includes('ppo_main'));
  if (pref) document.getElementById('agent').value = pref[0];
  newMatch();
})();
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8777)
    args = p.parse_args()
    print(f"Pişti GUI -> http://localhost:{args.port}")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
