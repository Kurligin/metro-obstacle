// Веб-прототип: выбор записи → обработка на сервере → просмотр кадров.
// Оси сцены — оси сообщения лидара (Z вверх); маркеры — те же, что у ноды (viz.py).
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
const fmt = (v, d = 1) =>
  v === null || v === undefined || !Number.isFinite(Number(v)) ? "—" : Number(v).toFixed(d);

// Цвета — те же токены, что в style.css.
const C = {
  fg: "#e8eaed",
  fg2: "#9aa0aa",
  fg3: "#8a909b",
  line: "#2a2e37",
  accent: "#4a8cff",
  ok: "#5aa86f",
  warn: "#c9a227",
  bad: "#c4595a",
};
const BOX = { red: 0xff3b3b, yellow: 0xe0b43a };
const FRAME_DT = 0.1; // запись 10 Гц — оценка шага, пока ряды не пришли
const BUDGET_MS = 100;

// Параметры URL: ?cam=cab|chase|overview — пресет камеры, ?color=height|intensity —
// раскраска облака, ?rec=1 — режим записи ролика (только вьюпорт, статус и шкала),
// ?frame=N — кадр, который показать после открытия задачи.
const params = new URLSearchParams(location.search);
const CAMS = ["cab", "chase", "overview"];
const REC = params.get("rec") === "1";
const initFrame = Number.parseInt(params.get("frame") || "", 10);

const state = {
  job: null, // статус задачи с сервера
  series: null,
  frame: 0,
  frameData: null, // последний показанный кадр
  playing: false,
  speed: 1,
  seq: 0, // номер последнего запроса кадра: ответы на устаревшие отбрасываются
  forward: null, // направление «вперёд» в осях лидара (по оси коридора)
  poll: null,
  jobs: [],
  cam: CAMS.includes(params.get("cam")) ? params.get("cam") : null, // null — прежний вид сзади
  color: params.get("color") === "intensity" ? "intensity" : "height",
  cloudBuf: null, // облако показанного кадра: перекрасить без повторной загрузки
};

// ------------------------------------------------------------------ API

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      msg = (await r.json()).error || msg;
    } catch {
      /* тело не JSON */
    }
    throw new Error(msg);
  }
  return r;
}

const getJson = async (path) => (await api(path)).json();

function showError(msg) {
  const el = $("error");
  el.textContent = msg || "";
  el.classList.toggle("hidden", !msg);
}
$("error").addEventListener("click", () => showError(""));

function setProgress(frac, text) {
  const el = $("progress");
  $("progress-text").textContent = frac === null ? "" : text;
  if (frac === null) {
    el.classList.add("hidden");
    return;
  }
  el.classList.remove("hidden");
  el.querySelector("i").style.width = `${Math.round(frac * 100)}%`;
}

// ------------------------------------------------------------------ форматирование

function fmtT(t) {
  if (t === null || t === undefined || !Number.isFinite(t)) return "—";
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function sizeText(bytes) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} ГБ`;
  return `${Math.max(1, Math.round(bytes / 1e6))} МБ`;
}

function plural(n, one, few, many) {
  const a = Math.abs(n) % 100;
  const b = a % 10;
  if (a > 10 && a < 20) return many;
  if (b === 1) return one;
  if (b >= 2 && b <= 4) return few;
  return many;
}

// ------------------------------------------------------------------ выбор записи

async function loadBags(selectId) {
  const { bags } = await getJson("/api/bags");
  const sel = $("bag-select");
  sel.innerHTML = "";
  if (!bags.length) sel.add(new Option("— каталог пуст —", ""));
  for (const b of bags) {
    const where = b.root === "uploads" ? "загружено: " : "";
    sel.add(new Option(`${where}${b.name} · ${sizeText(b.size)}`, b.id));
  }
  // по умолчанию — запись открытой задачи (если страницу открыли по ссылке на задачу)
  const want = selectId || (state.job && state.job.bag.id);
  if (want) sel.value = want;
}

function upload(file) {
  // XHR, а не fetch: нужен прогресс отправки (записи бывают в несколько ГБ).
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload?name=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        setProgress(e.loaded / e.total, `загрузка ${Math.round((100 * e.loaded) / e.total)}%`);
      }
    };
    xhr.onload = () => {
      let body = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* пусто */
      }
      if (xhr.status === 201) resolve(body);
      else reject(new Error(body.error || `загрузка: ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("загрузка оборвалась"));
    xhr.send(file);
  });
}

function setFilePicked(file) {
  const lbl = $("bag-file").closest("label");
  lbl.classList.toggle("picked", !!file);
  $("bag-file-name").textContent = file ? file.name : "Загрузить .db3 / .zip";
}

async function onRun(ev) {
  ev.preventDefault();
  showError("");
  const btn = $("run-btn");
  btn.disabled = true;
  try {
    let bag = $("bag-select").value;
    const file = $("bag-file").files[0];
    if (file) {
      const up = await upload(file);
      bag = up.bag;
      $("bag-file").value = "";
      setFilePicked(null);
      await loadBags(bag);
    }
    if (!bag) throw new Error("выберите запись или загрузите файл");
    const mf = parseInt($("max-frames").value, 10);
    const r = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bag, max_frames: Number.isFinite(mf) && mf > 0 ? mf : null }),
    });
    const { id } = await r.json();
    history.replaceState(null, "", `#job=${id}`);
    openJob(id);
  } catch (e) {
    setProgress(null);
    showError(e.message);
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------------ задачи

async function loadJobs() {
  try {
    const { jobs } = await getJson("/api/jobs");
    state.jobs = jobs;
  } catch {
    return; // список — второстепенное, ошибку не показываем
  }
  renderJobs();
}

function jobTime(id) {
  const m = /^\d{8}-(\d{2})(\d{2})/.exec(id);
  return m ? `${m[1]}:${m[2]}` : "";
}

function renderJobs() {
  const el = $("jobs");
  const jobs = [...state.jobs].reverse();
  if (!jobs.length) {
    el.innerHTML = `<div class="empty">Пока ничего не обработано</div>`;
    return;
  }
  const cur = state.job && state.job.id;
  el.innerHTML = jobs
    .map((j) => {
      const s = j.summary;
      let sc;
      let cls = "";
      if (j.state === "running" || j.state === "queued") {
        sc = j.total ? `${Math.round((100 * j.done) / j.total)}%` : "…";
        cls = "dim";
      } else if (j.state === "error") {
        sc = "ошибка";
        cls = "bad";
      } else if (s.first_alarm) {
        sc = `${fmt(s.first_alarm.distance)} м`;
        cls = "bad";
      } else {
        sc = "свободно";
        cls = "ok";
      }
      const sub = [
        `${j.total} ${plural(j.total, "кадр", "кадра", "кадров")}`,
        s.alarm_frames ? `тревога ${s.alarm_frames}` : "без тревог",
        jobTime(j.id),
      ]
        .filter(Boolean)
        .join(" · ");
      return `<button type="button" class="run${j.id === cur ? " on" : ""}" data-id="${esc(j.id)}">
        <span class="nm">${esc(j.bag.name)}</span><span class="sc ${cls}">${sc}</span>
        <span class="sub">${sub}</span>
      </button>`;
    })
    .join("");
}

$("jobs").addEventListener("click", (e) => {
  const b = e.target.closest(".run[data-id]");
  if (!b || (state.job && b.dataset.id === state.job.id)) return;
  history.replaceState(null, "", `#job=${b.dataset.id}`);
  openJob(b.dataset.id);
});

function resetJobView() {
  state.job = null;
  state.series = null;
  state.frame = 0;
  state.frameData = null;
  state.forward = null;
  state.cloudBuf = null;
  stopPlay();
  disposeGroup(groups.cloud);
  disposeGroup(groups.overlay);
  disposeGroup(groups.grid);
  groups.cloud = groups.overlay = groups.grid = null;
  labels.forEach((l) => l.el.remove());
  labels = [];
  $("view-empty").classList.remove("hidden");
  $("view-empty").querySelector(".empty-box").textContent = "Обработка…";
  $("view-cap").innerHTML = "";
  renderNow(null);
  renderRecStatus(null);
  renderSeries();
}

async function openJob(id) {
  clearTimeout(state.poll);
  resetJobView();
  let firstFrameShown = false;
  let lastSeries = 0;
  let lastJobs = 0;
  const tick = async () => {
    let st;
    try {
      st = await getJson(`/api/jobs/${id}`);
    } catch (e) {
      showError(e.message);
      setProgress(null);
      return;
    }
    if (!state.job) $("bag-select").value = st.bag.id; // открыли задачу по ссылке
    state.job = st;
    const running = st.state === "queued" || st.state === "running";
    if (st.state === "queued") setProgress(0, "в очереди…");
    else if (running) {
      setProgress(
        st.total ? st.done / st.total : 0,
        `${st.done} / ${st.total || "?"} кадров · ${fmt(st.summary.wall_s)} с`,
      );
    } else if (st.state === "error") {
      setProgress(null);
      showError(`Ошибка обработки: ${st.error}`);
    } else {
      setProgress(null);
    }
    renderRecord(st);
    renderMetrics(st);
    renderEvents(st);
    renderDownloads(st);
    const now = performance.now();
    if (!running || now - lastJobs > 1500) {
      lastJobs = now;
      loadJobs();
    }
    if (st.done > 0 && (!running || now - lastSeries > 1000)) {
      lastSeries = now;
      state.series = await getJson(`/api/jobs/${id}/series`);
      renderSeries();
      renderEvents(st);
      if (!firstFrameShown) {
        firstFrameShown = true;
        $("view-empty").classList.add("hidden");
        showFrame(0);
      }
    }
    if (!running && st.state === "done") {
      if (!st.done) {
        $("view-empty").querySelector(".empty-box").textContent = "В записи нет кадров облака";
        return;
      }
      const first = st.summary.first_alarm;
      const k = Number.isFinite(initFrame) ? initFrame : first ? first.frame : state.frame;
      showFrame(k, true);
      return;
    }
    if (running) state.poll = setTimeout(tick, 400);
  };
  tick();
}

// ------------------------------------------------------------------ левая панель

function renderRecord(st) {
  const s = st.summary;
  const frames =
    st.total && s.frames !== st.total ? `${s.frames} из ${st.total}` : `${st.total || s.frames}`;
  const rows = [
    ["Имя", esc(st.bag.name), ""],
    ["Кадров", frames, "mono"],
    ["Длительность", s.record_s > 0 ? `${fmt(s.record_s)} с` : "—", "mono"],
    ["Топик", `<span title="${esc(st.topic || "")}">${esc(st.topic || "—")}</span>`, "mono"],
  ];
  $("record").innerHTML =
    rows.map(([k, v, c]) => `<dt>${k}</dt><dd class="${c}">${v}</dd>`).join("") +
    `<dd class="path" title="${esc(st.bag.id)}">${esc(st.bag.id)}</dd>`;
  $("mode").textContent = st.mode || "default";
}

function blk(lbl, frac, val, cls = "", valCls = "") {
  const w = Math.max(0, Math.min(1, frac || 0)) * 100;
  return `<div class="blk"><span class="lbl">${lbl}</span>
    <span class="bar"><i class="${cls}" style="width:${w.toFixed(1)}%"></i></span>
    <span class="val ${valCls}">${val}</span></div>`;
}

function renderMetrics(st) {
  const s = st.summary;
  const n = $("metric-n");
  const sub = $("metric-sub");
  if (s.first_alarm) {
    n.textContent = fmt(s.first_alarm.distance);
    n.className = "n bad";
    $("metric-of").textContent = "м";
    sub.textContent = `кадр ${s.first_alarm.frame} · ${fmt(s.first_alarm.t, 2)} с · ближе всего ${fmt(s.min_distance)} м`;
  } else if (st.state === "done") {
    n.textContent = "нет";
    n.className = "n ok";
    $("metric-of").textContent = "";
    sub.textContent = "препятствий в габарите не найдено";
  } else {
    n.textContent = "—";
    n.className = "n";
    $("metric-of").textContent = "";
    sub.textContent = st.state === "error" ? "обработка прервалась" : "идёт обработка…";
  }
  const f = Math.max(s.frames, 1);
  $("alarm-blocks").innerHTML =
    blk("с тревогой", s.alarm_frames / f, `${s.alarm_frames} / ${s.frames}`, "bad") +
    blk("кандидаты", s.candidate_frames / f, `${s.candidate_frames} / ${s.frames}`, "warn");

  const b = (v) => (v === null ? 0 : v / BUDGET_MS);
  const over = (v) => (v !== null && v > BUDGET_MS ? "bad" : "");
  const c = s.core_ms;
  const fm = s.frame_ms;
  $("time-blocks").innerHTML =
    blk("ядро, мин", b(c.min), fmt(c.min), over(c.min), over(c.min)) +
    blk("ядро, мед.", b(c.median), fmt(c.median), over(c.median), over(c.median)) +
    blk("ядро, p95", b(c.p95), fmt(c.p95), over(c.p95), over(c.p95)) +
    blk("кадр, мед.", b(fm.median), fmt(fm.median), over(fm.median), over(fm.median)) +
    blk("кадр, p95", b(fm.p95), fmt(fm.p95), over(fm.p95), over(fm.p95));
  const speed = [
    ["Обработка", `${fmt(s.wall_s)} с`],
    ["Длительность записи", s.record_s > 0 ? `${fmt(s.record_s)} с` : "—"],
    [
      "Быстрее реального",
      s.record_s > 0 && s.wall_s > 0 ? `×${fmt(s.record_s / Math.max(s.wall_s, 1e-3))}` : "—",
    ],
  ];
  $("speed").innerHTML = speed.map(([k, v]) => `<dt>${k}</dt><dd class="mono">${v}</dd>`).join("");
}

function renderEmptyLeft() {
  $("record").innerHTML =
    `<dt>Имя</dt><dd>—</dd><dt>Кадров</dt><dd>—</dd><dt>Длительность</dt><dd>—</dd>`;
  $("alarm-blocks").innerHTML = "";
  $("time-blocks").innerHTML = `<div class="empty">появится после обработки</div>`;
  $("speed").innerHTML = "";
  $("events").innerHTML = `<tr><td colspan="4" class="empty">—</td></tr>`;
}

// ------------------------------------------------------------------ правая панель

function statusOf(frame) {
  const cand = frame.boxes.filter((b) => !b.confirmed);
  if (frame.obstacle) return { cls: "alarm", text: `ПРЕПЯТСТВИЕ ${fmt(frame.distance)} м` };
  if (frame.level === 1 && cand.length)
    return {
      cls: "candidate",
      text: `КАНДИДАТ ${fmt(Math.min(...cand.map((b) => b.distance)))} м`,
    };
  return { cls: "clear", text: "ПУТЬ СВОБОДЕН" };
}

function renderNow(frame) {
  const title = $("now-title");
  if (!frame) {
    title.textContent = "Сейчас";
    $("now").innerHTML = [
      ["Статус", `<span class="status">нет данных</span>`],
      ["Дистанция", "—"],
      ["Уверенность", "—"],
      ["Видимая дальность", "—"],
      ["Время ядра", "—"],
      ["Кадр целиком", "—"],
      ["Калибровка", "—"],
    ]
      .map(
        ([k, v]) => `<div class="r"><span class="k">${k}</span><span class="v">${v}</span></div>`,
      )
      .join("");
    $("objects").innerHTML = `<tr><td colspan="5" class="empty">—</td></tr>`;
    $("obj-count").textContent = "";
    return;
  }
  title.textContent = `Сейчас · кадр ${frame.frame} · ${fmt(frame.t, 2)} с`;
  const st = statusOf(frame);
  const conf = frame.confidence ? fmt(frame.confidence, 2) : "—";
  const over = (v) => (v !== null && v > BUDGET_MS ? " bad" : "");
  const rows = [
    ["Статус", `<span class="status ${st.cls}">${st.text}</span>`, "status-cell"],
    ["Дистанция", frame.obstacle ? `${fmt(frame.distance)} м` : "—", frame.obstacle ? "bad" : ""],
    ["Уверенность", conf, ""],
    ["Видимая дальность", `${fmt(frame.visible_range, 0)} м`, ""],
    ["Время ядра", `${fmt(frame.core_ms)} мс`, over(frame.core_ms)],
    ["Кадр целиком", `${fmt(frame.processing_ms)} мс`, over(frame.processing_ms)],
    [
      "Калибровка",
      frame.calibrated ? `<span class="ok">готова</span>` : `<span class="warn">идёт…</span>`,
      "",
    ],
  ];
  $("now").innerHTML = rows
    .map(
      ([k, v, c]) =>
        `<div class="r"><span class="k">${k}</span><span class="v ${c}">${v}</span></div>`,
    )
    .join("");

  const boxes = frame.boxes;
  $("obj-count").textContent = boxes.length ? `${boxes.length}` : "";
  $("objects").innerHTML =
    boxes
      .map(
        (b) => `<tr>
        <td class="${b.confirmed ? "bad" : "warn"}">${fmt(b.distance)}</td>
        <td>${fmt(b.lateral, 2)}</td>
        <td>${fmt(b.width)}×${fmt(b.height)}×${fmt(b.length)}</td>
        <td>${b.points}</td>
        <td>${b.confirmed ? '<span class="tag bad">подтв.</span>' : '<span class="tag warn">канд.</span>'}</td>
      </tr>`,
      )
      .join("") || `<tr><td colspan="5" class="empty">нет объектов в габарите</td></tr>`;
}

function renderEvents(st) {
  const tb = $("events");
  if (!st.events.length) {
    tb.innerHTML = `<tr><td colspan="4" class="empty">${st.state === "done" ? "тревог нет" : "—"}</td></tr>`;
  } else {
    const t = state.series ? state.series.t : null;
    tb.innerHTML = st.events
      .map((e, i) => {
        const time = t && t[e.end] !== undefined ? `${fmt(t[e.start])}–${fmt(t[e.end])}` : "—";
        return `<tr class="click" data-frame="${e.start}" data-i="${i}">
          <td>${e.start}–${e.end} <span class="tag bad">${e.frames}</span></td>
          <td>${time}</td>
          <td>${fmt(e.first_distance)}→${fmt(e.min_distance)}</td>
          <td>${fmt(e.width)}×${fmt(e.height)}×${fmt(e.length)}</td>
        </tr>`;
      })
      .join("");
  }
  const r = $("resets");
  r.classList.toggle("hidden", !st.resets.length);
  if (st.resets.length) {
    r.textContent = `Сбросов ядра (скачок времени в записи): ${st.resets.length} — кадры ${st.resets
      .map((x) => x.frame)
      .join(", ")}`;
  }
  markEvent();
}

function markEvent() {
  if (!state.job) return;
  for (const tr of $("events").querySelectorAll("tr[data-i]")) {
    const e = state.job.events[Number(tr.dataset.i)];
    tr.classList.toggle("cur", state.frame >= e.start && state.frame <= e.end);
  }
}

function renderDownloads(st) {
  const a = $("dl-results");
  const done = st && st.state === "done";
  a.classList.toggle("disabled", !done);
  a.href = done ? `/api/jobs/${st.id}/results.jsonl` : "#";
  const b = $("dl-mcap");
  if (!st) {
    b.disabled = true;
    return;
  }
  const m = st.mcap;
  b.disabled = !done || m.state === "queued" || m.state === "running";
  if (m.state === "running" || m.state === "queued") {
    b.textContent = `Сборка .mcap: ${m.done} / ${st.total}`;
  } else if (m.state === "done") {
    b.textContent = "Скачать .mcap";
  } else if (m.state === "error") {
    b.textContent = ".mcap: ошибка, повторить";
  } else {
    b.textContent = ".mcap для Lichtblick";
  }
}

async function onMcap() {
  const st = state.job;
  if (!st) return;
  if (st.mcap.state === "done") {
    window.location.href = `/api/jobs/${st.id}/mcap`;
    return;
  }
  try {
    await api(`/api/jobs/${st.id}/mcap`, { method: "POST" });
  } catch (e) {
    showError(e.message);
    return;
  }
  const poll = async () => {
    const s = await getJson(`/api/jobs/${st.id}`);
    if (!state.job || state.job.id !== s.id) return;
    state.job = s;
    renderDownloads(s);
    if (s.mcap.state === "queued" || s.mcap.state === "running") setTimeout(poll, 500);
    else if (s.mcap.state === "error") showError(`MCAP: ${s.mcap.error}`);
  };
  poll();
}

// ------------------------------------------------------------------ таймлайн

const tracksCv = $("tracks-canvas");
const LX = 150;
const RX = 14;
const ROWS = [
  { k: "axis", h: 16 },
  { k: "status", h: 12, lbl: "статус" },
  { k: "dist", h: 36, lbl: "дистанция, м" },
  { k: "ms", h: 36, lbl: "время кадра, мс" },
  { k: "vis", h: 36, lbl: "видимая дальность, м" },
];

function totalFrames() {
  const n = state.series ? state.series.t.length : 0;
  return Math.max(state.job && state.job.total ? state.job.total : n, n);
}

function frameDt() {
  const t = state.series ? state.series.t : [];
  return t.length > 1 ? t[t.length - 1] / (t.length - 1) : FRAME_DT;
}

function frameX(k, W) {
  const total = totalFrames();
  return LX + (total > 1 ? k / (total - 1) : 0.5) * (W - LX - RX);
}

function niceMax(v, floor) {
  const steps = [10, 20, 25, 50, 100, 150, 200, 250, 300, 400, 500, 1000];
  const x = Math.max(v, floor);
  return steps.find((s) => s >= x) || Math.ceil(x / 500) * 500;
}

function pct(arr, q) {
  const v = arr.filter((x) => x !== null && Number.isFinite(x)).sort((a, b) => a - b);
  return v.length ? v[Math.min(v.length - 1, Math.floor(q * (v.length - 1)))] : null;
}

function renderSeries() {
  const n = state.series ? state.series.t.length : 0;
  $("play").disabled = n === 0;
  updateClock();
  drawTracks();
}

function drawTracks() {
  const W = tracksCv.clientWidth;
  const H = tracksCv.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  tracksCv.width = Math.round(W * dpr);
  tracksCv.height = Math.round(H * dpr);
  const ctx = tracksCv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!W) return;
  const s = state.series;
  const n = s ? s.t.length : 0;
  const total = totalFrames();
  const X = (k) => frameX(k, W);
  const pw = W - LX - RX;
  const sum = ROWS.reduce((a, r) => a + r.h + 4, 0);
  const kk = (H - 6) / sum;
  let y = 4;
  ctx.textBaseline = "middle";
  const ui = "11px Inter, system-ui, sans-serif";
  const mono = "10px 'JetBrains Mono', monospace";
  ctx.font = ui;

  for (const row of ROWS) {
    const h = row.h * kk;
    if (row.lbl) {
      ctx.fillStyle = C.fg2;
      ctx.textAlign = "right";
      ctx.font = ui;
      ctx.fillText(row.lbl, LX - 10, y + h / 2);
    }
    if (row.k === "axis") {
      if (total > 0) {
        const dt = frameDt();
        const tMax = (total - 1) * dt;
        const step = [1, 2, 5, 10, 15, 30, 60, 120, 300].find((st) => tMax / st <= 12) || 600;
        ctx.font = mono;
        ctx.textAlign = "center";
        for (let t = 0; t <= tMax + 1e-6; t += step) {
          const x = LX + (tMax > 0 ? t / tMax : 0) * pw;
          ctx.fillStyle = C.fg3;
          ctx.textAlign = x > W - RX - 20 ? "right" : x < LX + 20 ? "left" : "center";
          ctx.fillText(fmtT(t).replace(/\.0$/, ""), x, y + h / 2);
          ctx.fillStyle = "rgba(255,255,255,0.04)";
          ctx.fillRect(x, y + h, 1, H);
        }
      }
    } else if (row.k === "status") {
      ctx.fillStyle = "rgba(255,255,255,0.03)";
      ctx.fillRect(LX, y, pw, h);
      if (n) {
        const bw = pw / Math.max(total - 1, 1);
        // слить соседние кадры одного уровня в один прямоугольник
        let k0 = 0;
        for (let k = 1; k <= n; k++) {
          if (k === n || s.level[k] !== s.level[k0]) {
            const lv = s.level[k0];
            ctx.fillStyle =
              lv === 2
                ? "rgba(196,89,90,0.85)"
                : lv === 1
                  ? "rgba(201,162,39,0.75)"
                  : "rgba(90,168,111,0.55)";
            const x0 = Math.max(LX, X(k0) - bw / 2);
            const x1 = Math.min(W - RX, X(k - 1) + bw / 2);
            ctx.fillRect(x0, y, Math.max(1, x1 - x0), h);
            k0 = k;
          }
        }
      }
    } else {
      ctx.fillStyle = "rgba(255,255,255,0.025)";
      ctx.fillRect(LX, y, pw, h);
      let max = 100;
      if (n && row.k === "dist") {
        const d = s.nearest.filter((v) => v !== null);
        max = niceMax(d.length ? Math.max(...d) * 1.1 : 50, 20);
      } else if (n && row.k === "ms") {
        max = niceMax((pct(s.frame_ms || s.core_ms, 0.98) || 0) * 1.2, 150);
      } else if (n && row.k === "vis") {
        const v = (s.visible_range || []).filter((x) => x !== null);
        max = niceMax(v.length ? Math.max(...v) * 1.05 : 100, 50);
      }
      const Yv = (v) => y + h - (Math.min(max, Math.max(0, v)) / max) * h;
      const line = (arr, color, width = 1.3, dash = null) => {
        if (!arr) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        if (dash) ctx.setLineDash(dash);
        ctx.beginPath();
        let on = false;
        for (let k = 0; k < n; k++) {
          const v = arr[k];
          if (v === null || v === undefined) {
            on = false;
            continue;
          }
          if (!on) {
            ctx.moveTo(X(k), Yv(v));
            on = true;
          } else ctx.lineTo(X(k), Yv(v));
        }
        ctx.stroke();
        ctx.setLineDash([]);
      };
      const hline = (v, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(LX, Math.round(Yv(v)) + 0.5);
        ctx.lineTo(W - RX, Math.round(Yv(v)) + 0.5);
        ctx.stroke();
        ctx.setLineDash([]);
      };
      if (n && row.k === "dist") {
        // кандидаты — жёлтые точки, подтверждённое препятствие — красная линия
        ctx.fillStyle = "rgba(201,162,39,0.85)";
        for (let k = 0; k < n; k++) {
          if (s.nearest[k] !== null && !s.obstacle[k])
            ctx.fillRect(X(k) - 1, Yv(s.nearest[k]) - 1, 2, 2);
        }
        line(s.distance, C.bad, 1.6);
        ctx.fillStyle = C.bad;
        for (let k = 0; k < n; k++) {
          if (s.distance[k] !== null && (k === 0 || s.distance[k - 1] === null)) {
            ctx.beginPath();
            ctx.arc(X(k), Yv(s.distance[k]), 2.2, 0, 2 * Math.PI);
            ctx.fill();
          }
        }
      } else if (n && row.k === "ms") {
        hline(BUDGET_MS, "rgba(196,89,90,0.7)");
        line(s.core_ms, "rgba(154,160,170,0.7)", 1);
        line(s.frame_ms, C.accent, 1.3);
        ctx.font = mono;
        ctx.textAlign = "right";
        ctx.fillStyle = C.fg3;
        ctx.fillText("кадр", W - RX - 70, y + 7);
        ctx.fillStyle = C.accent;
        ctx.fillRect(W - RX - 66, y + 6, 10, 2);
        ctx.fillStyle = C.fg3;
        ctx.fillText("ядро", W - RX - 16, y + 7);
        ctx.fillStyle = "rgba(154,160,170,0.8)";
        ctx.fillRect(W - RX - 12, y + 6, 10, 2);
        ctx.fillStyle = "rgba(196,89,90,0.9)";
        ctx.textAlign = "left";
        ctx.fillText(`бюджет ${BUDGET_MS}`, LX + 40, Yv(BUDGET_MS) - 6);
      } else if (n && row.k === "vis") {
        line(s.visible_range, C.ok, 1.3);
      }
      if (n) {
        ctx.fillStyle = C.fg3;
        ctx.textAlign = "left";
        ctx.font = mono;
        ctx.fillText(String(max), LX + 3, y + 7);
      }
    }
    y += h + 4 * kk;
  }

  // курсор текущего кадра по всем дорожкам
  if (n) {
    const cx = Math.round(X(state.frame)) + 0.5;
    ctx.strokeStyle = C.fg;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 4);
    ctx.lineTo(cx, H);
    ctx.stroke();
    ctx.fillStyle = C.fg;
    ctx.beginPath();
    ctx.moveTo(cx - 5, 1);
    ctx.lineTo(cx + 5, 1);
    ctx.lineTo(cx, 7);
    ctx.fill();
  } else {
    ctx.fillStyle = C.fg3;
    ctx.textAlign = "center";
    ctx.font = ui;
    ctx.fillText("шкала появится после обработки записи", LX + pw / 2, H / 2);
  }
}

function frameAt(clientX) {
  const rect = tracksCv.getBoundingClientRect();
  const W = rect.width;
  const total = totalFrames();
  const frac = (clientX - rect.left - LX) / (W - LX - RX);
  const k = Math.round(Math.min(Math.max(frac, 0), 1) * Math.max(total - 1, 0));
  return Math.min(k, state.series.t.length - 1);
}

let dragging = false;
tracksCv.addEventListener("pointerdown", (e) => {
  if (!state.series || !state.series.t.length) return;
  dragging = true;
  tracksCv.setPointerCapture(e.pointerId);
  stopPlay();
  showFrame(frameAt(e.clientX));
});
tracksCv.addEventListener("pointermove", (e) => {
  if (dragging) showFrame(frameAt(e.clientX));
});
tracksCv.addEventListener("pointerup", () => {
  dragging = false;
});

function updateClock() {
  const s = state.series;
  const n = s ? s.t.length : 0;
  const total = totalFrames();
  if (!n) {
    $("time").textContent = "0:00.0 / 0:00.0";
    $("frame-no").textContent = "кадр — / —";
    return;
  }
  const dur = state.job && state.job.state === "done" ? s.t[n - 1] : (total - 1) * frameDt();
  $("time").textContent = `${fmtT(s.t[state.frame])} / ${fmtT(dur)}`;
  $("frame-no").textContent = `кадр ${state.frame} / ${total - 1}`;
}

// ------------------------------------------------------------------ 3D-вид

const view = $("view");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setClearColor(0x0b0d11);
view.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 3000);
camera.up.set(0, 0, 1);
camera.position.set(-15, 0, 10);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.15;
controls.target.set(30, 0, 0);

const groups = { cloud: null, overlay: null, grid: null };
let labels = []; // {el, pos: Vector3}

function resize() {
  const w = view.clientWidth;
  const h = view.clientHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = `${w}px`;
  renderer.domElement.style.height = `${h}px`;
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);
new ResizeObserver(drawTracks).observe($("tracks"));

function viewFromBehind() {
  const f = state.forward || new THREE.Vector3(1, 0, 0);
  scene.fog = null;
  // внутри тоннеля: чуть сзади и выше лидара, взгляд вдоль пути
  setFov(55);
  camera.position.copy(f.clone().multiplyScalar(-9)).add(new THREE.Vector3(0, 0, 4.5));
  controls.target.copy(f.clone().multiplyScalar(30)).add(new THREE.Vector3(0, 0, -2.5));
  controls.update();
}

function setFov(fov) {
  camera.fov = fov;
  camera.near = 0.3;
  camera.updateProjectionMatrix();
}

// Ось пути по дальности вдоль «вперёд»: точка оси на расстоянии s (м) от лидара.
// Вне прослеженной оси — продолжение крайнего отрезка; без оси — прямая по f на уровне полотна.
function axisAt(s) {
  const f = state.forward || new THREE.Vector3(1, 0, 0);
  const axis = state.frameData && state.frameData.corridor && state.frameData.corridor.axis;
  if (!axis || axis.length < 2)
    return f
      .clone()
      .multiplyScalar(s)
      .add(new THREE.Vector3(0, 0, -1.5));
  const d = axis.map((p) => p[0] * f.x + p[1] * f.y);
  let i = 0;
  while (i < axis.length - 2 && d[i + 1] < s) i++;
  const u = (s - d[i]) / Math.max(d[i + 1] - d[i], 1e-6);
  const p = axis[i];
  const q = axis[i + 1];
  return new THREE.Vector3(
    p[0] + (q[0] - p[0]) * u,
    p[1] + (q[1] - p[1]) * u,
    p[2] + (q[2] - p[2]) * u,
  );
}

const CAM_H = 1.1; // высота «глаза» кабины над осью пути (головками рельсов), м

// Пресеты камеры. Все считаются от оси пути текущего кадра, «вперёд» — как у вида сзади.
function applyCam(name) {
  if (!CAMS.includes(name)) {
    viewFromBehind();
    return;
  }
  const f = state.forward || new THREE.Vector3(1, 0, 0);
  const side = new THREE.Vector3(-f.y, f.x, 0); // влево по ходу
  const up = new THREE.Vector3(0, 0, 1);
  const eye = axisAt(0).add(up.clone().multiplyScalar(CAM_H));
  // дымка по глубине: в видах изнутри тоннеля даль темнее, ближнее — ярче
  scene.fog = name === "overview" ? null : new THREE.Fog(0x0b0d11, 20, 150);
  if (name === "cab") {
    // кабина машиниста: из точки лидара вдоль пути, рельсы уходят в перспективу
    setFov(58);
    camera.position.copy(eye);
    controls.target.copy(axisAt(60).add(up.clone().multiplyScalar(CAM_H * 0.8)));
  } else if (name === "chase") {
    // за поездом: позади и выше лидара, взгляд чуть вниз
    setFov(55);
    camera.position.copy(eye).add(f.clone().multiplyScalar(-7)).add(up.clone().multiplyScalar(3.5));
    controls.target.copy(axisAt(40).add(up.clone().multiplyScalar(-7)));
  } else {
    // сверху-сбоку: коридор вдоль пути на 100+ м; потолок тоннеля скрыт (см. cloudObject)
    setFov(45);
    const c = axisAt(50);
    controls.target.copy(c).add(up.clone().multiplyScalar(0.5));
    const el = (52 * Math.PI) / 180; // угол сверху: взгляд через срезанную стенку в коридор
    camera.position
      .copy(controls.target)
      .add(side.clone().multiplyScalar(-62 * Math.cos(el)))
      .add(f.clone().multiplyScalar(-4))
      .add(up.clone().multiplyScalar(62 * Math.sin(el)));
  }
  controls.update();
}

function setCam(name) {
  state.cam = name;
  for (const b of $("campresets").querySelectorAll("button[data-cam]"))
    b.classList.toggle("on", b.dataset.cam === name);
  syncUrl();
  if (state.cloudBuf && state.frameData) rebuildCloud();
  applyCam(name);
}

function syncUrl() {
  const u = new URL(location.href);
  if (state.cam) u.searchParams.set("cam", state.cam);
  else u.searchParams.delete("cam");
  if (state.color === "intensity") u.searchParams.set("color", "intensity");
  else u.searchParams.delete("color");
  history.replaceState(null, "", u);
}

function fitView() {
  if (!groups.cloud) return;
  const box = new THREE.Box3().setFromObject(groups.cloud);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  const f = state.forward || new THREE.Vector3(1, 0, 0);
  // косой вид сверху-сзади: облако целиком
  controls.target.copy(center);
  camera.position
    .copy(center)
    .add(f.clone().multiplyScalar(-size * 0.45))
    .add(new THREE.Vector3(0, 0, size * 0.45));
  controls.update();
}

function zoom(k) {
  const d = camera.position.clone().sub(controls.target);
  camera.position.copy(controls.target).add(d.multiplyScalar(k));
  controls.update();
}

// Спокойная шкала без радуги: тёмно-синий → серо-голубой → светлый.
const RAMP = [
  [0.0, [0.09, 0.13, 0.3]],
  [0.45, [0.33, 0.45, 0.6]],
  [1.0, [0.86, 0.9, 0.95]],
];
const COL_GAUGE = [0.98, 0.84, 0.45]; // точка внутри габарита выше полотна
const COL_OBJ = [1.0, 0.18, 0.16]; // точка подтверждённого объекта
const COL_RAIL = [0.72, 0.8, 0.9]; // головки рельсов — светлые линии в перспективу
const RAIL_HALF = 0.76; // полуколея 1520 мм, м
const RAIL_CLEAR = 0.25; // выше головок рельсов на столько — уже «в габарите», м
const CHASE_CUT = 22; // до скольких метров вперёд срезать свод в виде «за поездом»
const BOX_MARGIN = 0.15; // запас рамки объекта при отборе его точек, м

function ramp(t) {
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, a] = RAMP[i - 1];
      const [t1, b] = RAMP[i];
      const u = (t - t0) / (t1 - t0);
      return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
    }
  }
  return RAMP[RAMP.length - 1][1];
}

// Геометрия коридора для раскраски: дальность вдоль «вперёд», ось, полуширина, высота.
function corridorTable(frame) {
  const cor = frame && frame.corridor;
  if (!cor || cor.axis.length < 2 || cor.edges.length < 3) return null;
  const f = state.forward || new THREE.Vector3(1, 0, 0);
  const [left, right, leftUp] = cor.edges;
  const n = Math.min(cor.axis.length, left.length, right.length);
  const d = new Float64Array(n);
  const half = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = cor.axis[i];
    d[i] = p[0] * f.x + p[1] * f.y;
    half[i] = Math.hypot(left[i][0] - right[i][0], left[i][1] - right[i][1]) / 2;
  }
  return { f, axis: cor.axis, d, half, n, gauge: leftUp[0][2] - left[0][2] };
}

// Высота над полотном и смещение от оси для точки; вне оси — null.
function corridorPos(tb, x, y, z) {
  const s = x * tb.f.x + y * tb.f.y;
  const { d, axis, n } = tb;
  let lo = 0;
  let hi = n - 1;
  if (s <= d[0]) hi = 1;
  else if (s >= d[n - 1]) lo = n - 2;
  else {
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (d[mid] <= s) lo = mid;
      else hi = mid;
    }
  }
  const i = lo;
  const u = (s - d[i]) / Math.max(d[i + 1] - d[i], 1e-6);
  const p = axis[i];
  const q = axis[i + 1];
  const ax = p[0] + (q[0] - p[0]) * u;
  const ay = p[1] + (q[1] - p[1]) * u;
  const az = p[2] + (q[2] - p[2]) * u;
  const inside = s >= 1 && s <= d[n - 1];
  const lat = (x - ax) * -tb.f.y + (y - ay) * tb.f.x;
  const half = tb.half[Math.min(Math.max(u < 0.5 ? i : i + 1, 0), n - 1)];
  return { h: z - az, lat, half, inside };
}

function inBoxes(boxes, x, y, z) {
  for (const b of boxes) {
    const dx = x - b.center[0];
    const dy = y - b.center[1];
    const c = Math.cos(-b.yaw);
    const sn = Math.sin(-b.yaw);
    const lx = dx * c - dy * sn;
    const ly = dx * sn + dy * c;
    if (
      Math.abs(lx) <= b.size[0] / 2 + BOX_MARGIN &&
      Math.abs(ly) <= b.size[1] / 2 + BOX_MARGIN &&
      Math.abs(z - b.center[2]) <= b.size[2] / 2 + BOX_MARGIN
    )
      return true;
  }
  return false;
}

// Слой точек. С перспективой (world > 0) ближние точки крупнее дальних — глубина читается
// сразу; размер на экране зажат в [minPx, maxPx], чтобы дальние не пропадали.
function pointsLayer(pos, col, n, { px, world = 0, maxPx = 8, fog = true }) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos.subarray(0, 3 * n), 3));
  geom.setAttribute("color", new THREE.BufferAttribute(col.subarray(0, 3 * n), 3));
  const dpr = renderer.getPixelRatio();
  const mat = new THREE.PointsMaterial({
    size: world > 0 ? world : px * dpr,
    sizeAttenuation: world > 0,
    vertexColors: true,
    fog,
  });
  if (world > 0) {
    mat.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader.replace(
        "#include <logdepthbuf_vertex>",
        `gl_PointSize = clamp(gl_PointSize, ${(px * dpr).toFixed(2)}, ${(maxPx * dpr).toFixed(2)});
        #include <logdepthbuf_vertex>`,
      );
    };
  }
  return new THREE.Points(geom, mat);
}

// Облако кадра: фон — по высоте над полотном (или по интенсивности) в спокойной гамме;
// точки в габарите выше полотна — светлым тёплым, точки подтверждённого объекта — красным.
function cloudObject(buf, frame) {
  const pts = new Float32Array(buf);
  const n = pts.length / 4;
  const tb = corridorTable(frame);
  const boxes = (frame.boxes || []).filter((b) => b.confirmed);
  const gauge = tb && tb.gauge > 0.5 ? tb.gauge : 3.5;
  // потолок тоннеля закрывает коридор сверху — в видах снаружи его не рисуем
  // (за поездом — только ближний, до CHASE_CUT м: дальний свод держит «тоннель» в кадре)
  const cut = state.cam === "overview" || state.cam === "chase" ? gauge + 0.9 : Infinity;
  const cutTo = state.cam === "chase" ? CHASE_CUT : Infinity;
  const f = state.forward || new THREE.Vector3(1, 0, 0);
  let lo = 0;
  let span = 1;
  if (state.color === "intensity") {
    // нормировка по 2–98 перцентилю: у разных лидаров разная шкала интенсивности
    const sorted = new Float32Array(n);
    for (let i = 0; i < n; i++) sorted[i] = pts[4 * i + 3];
    sorted.sort();
    lo = sorted[Math.floor(0.02 * (n - 1))] || 0;
    const hi = sorted[Math.floor(0.98 * (n - 1))] || 1;
    span = hi > lo ? hi - lo : 1;
  }
  const L = [0, 1, 2].map(() => ({
    pos: new Float32Array(n * 3),
    col: new Float32Array(n * 3),
    k: 0,
  }));
  for (let i = 0; i < n; i++) {
    const x = pts[4 * i];
    const y = pts[4 * i + 1];
    const z = pts[4 * i + 2];
    const cp = tb ? corridorPos(tb, x, y, z) : null;
    const h = cp ? cp.h : z + 1.5;
    if (h > cut && x * f.x + y * f.y < cutTo) continue;
    let layer = 0;
    let rgb;
    if (boxes.length && inBoxes(boxes, x, y, z)) {
      layer = 2;
      rgb = COL_OBJ;
    } else if (cp && cp.inside && Math.abs(cp.lat) <= cp.half && h > RAIL_CLEAR && h < gauge) {
      layer = 1;
      rgb = COL_GAUGE;
    } else if (
      cp &&
      cp.inside &&
      Math.abs(h) < 0.15 &&
      Math.abs(Math.abs(cp.lat) - RAIL_HALF) < 0.12
    ) {
      rgb = COL_RAIL;
    } else if (state.color === "intensity") {
      rgb = ramp(Math.sqrt(Math.min(Math.max((pts[4 * i + 3] - lo) / span, 0), 1)));
    } else {
      rgb = ramp(Math.min(Math.max((h + 0.4) / (gauge + 2.5), 0), 1));
    }
    const o = L[layer];
    const j = 3 * o.k++;
    o.pos[j] = x;
    o.pos[j + 1] = y;
    o.pos[j + 2] = z;
    o.col[j] = rgb[0];
    o.col[j + 1] = rgb[1];
    o.col[j + 2] = rgb[2];
  }
  const g = new THREE.Group();
  const base = REC ? 2 : 1.8;
  const opts = [
    { px: base, world: 0.07, maxPx: 4.5 },
    { px: base + 1.4, world: 0.1, maxPx: 6, fog: false },
    { px: base + 4, fog: false },
  ];
  L.forEach((o, i) => {
    if (o.k) g.add(pointsLayer(o.pos, o.col, o.k, opts[i]));
  });
  return g;
}

function rebuildCloud() {
  disposeGroup(groups.cloud);
  disposeGroup(groups.overlay);
  groups.cloud = cloudObject(state.cloudBuf, state.frameData);
  groups.overlay = overlayObject(state.frameData);
  scene.add(groups.cloud, groups.overlay);
}

function polyline(points, color, opacity = 1) {
  const geom = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p)));
  const mat = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
  return new THREE.Line(geom, mat);
}

function overlayObject(frame) {
  const g = new THREE.Group();
  const levelColor = new THREE.Color(frame.color[0], frame.color[1], frame.color[2]);
  const cor = frame.corridor;
  if (cor) {
    for (const e of cor.edges) g.add(polyline(e, levelColor, 0.9));
    const seg = [];
    // из кабины ближняя рамка — «дверной проём» на весь кадр: её не рисуем
    const near = state.cam === "cab" ? 9 : 0;
    for (const [a, b] of cor.sections) {
      if (Math.hypot(a[0], a[1]) >= near)
        seg.push(new THREE.Vector3(...a), new THREE.Vector3(...b));
    }
    const sg = new THREE.BufferGeometry().setFromPoints(seg);
    g.add(
      new THREE.LineSegments(
        sg,
        new THREE.LineBasicMaterial({ color: levelColor, transparent: true, opacity: 0.45 }),
      ),
    );
    // полупрозрачная «труба» габарита: полотно, стенки и верх между кромками
    const [left, right, leftUp, rightUp] = cor.edges;
    const strip = (p, q, opacity) => {
      const tri = [];
      for (let i = 0; i + 1 < Math.min(p.length, q.length); i++) {
        tri.push(...p[i], ...q[i], ...p[i + 1], ...p[i + 1], ...q[i], ...q[i + 1]);
      }
      const fg = new THREE.BufferGeometry();
      fg.setAttribute("position", new THREE.Float32BufferAttribute(tri, 3));
      g.add(
        new THREE.Mesh(
          fg,
          new THREE.MeshBasicMaterial({
            color: levelColor,
            transparent: true,
            opacity,
            side: THREE.DoubleSide,
            depthWrite: false,
          }),
        ),
      );
    };
    strip(left, right, 0.14);
    if (leftUp && rightUp) {
      strip(left, leftUp, 0.045);
      strip(right, rightUp, 0.045);
      strip(leftUp, rightUp, 0.025);
    }
  }
  labels.forEach((l) => l.el.remove());
  labels = [];
  for (const b of frame.boxes) {
    const color = b.confirmed ? BOX.red : BOX.yellow;
    const geom = new THREE.BoxGeometry(...b.size);
    const box = new THREE.Group();
    box.add(
      new THREE.LineSegments(new THREE.EdgesGeometry(geom), new THREE.LineBasicMaterial({ color })),
    );
    box.add(
      new THREE.Mesh(
        geom,
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18, depthWrite: false }),
      ),
    );
    box.position.set(...b.center);
    box.rotation.z = b.yaw;
    g.add(box);
    const el = document.createElement("div");
    el.className = `label3d${b.confirmed ? "" : " candidate"}`;
    el.textContent = b.label;
    $("labels").append(el);
    labels.push({
      el,
      pos: new THREE.Vector3(b.center[0], b.center[1], b.center[2] + b.size[2] / 2 + 0.6),
    });
  }
  // лидар — точка отсчёта дистанций
  const lidar = new THREE.Mesh(
    new THREE.SphereGeometry(0.07, 12, 8),
    new THREE.MeshBasicMaterial({ color: 0x4a8cff }),
  );
  g.add(lidar);
  return g;
}

function gridObject(frame) {
  // приглушённая сетка 10 м на уровне полотна (по оси коридора), вдоль пути
  const axis = frame.corridor && frame.corridor.axis;
  const z = axis && axis.length ? axis[0][2] : -1.5;
  const grid = new THREE.GridHelper(400, 40, 0x1f232b, 0x171a20);
  grid.rotation.x = Math.PI / 2; // GridHelper лежит в XZ — повернуть в XY
  grid.position.set(0, 0, z - 0.05);
  if (state.forward) grid.rotation.y = Math.atan2(state.forward.y, state.forward.x);
  grid.material.depthWrite = false;
  return grid;
}

function disposeGroup(obj) {
  if (!obj) return;
  scene.remove(obj);
  obj.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
}

function forwardOf(frame) {
  const axis = frame.corridor && frame.corridor.axis;
  if (!axis || axis.length < 2) return null;
  const a = axis[0];
  const b = axis[Math.min(axis.length - 1, 20)];
  const v = new THREE.Vector3(b[0] - a[0], b[1] - a[1], 0);
  return v.length() > 1 ? v.normalize() : null;
}

async function showFrame(k, recenter = false) {
  if (!state.job || !state.series || !state.series.t.length) return;
  k = Math.min(Math.max(k, 0), state.series.t.length - 1);
  state.frame = k;
  updateClock();
  drawTracks();
  markEvent();
  const seq = ++state.seq;
  const id = state.job.id;
  try {
    const [frame, buf] = await Promise.all([
      getJson(`/api/jobs/${id}/frames/${k}`),
      api(`/api/jobs/${id}/frames/${k}/cloud`).then((r) => r.arrayBuffer()),
    ]);
    if (seq !== state.seq || !state.job || state.job.id !== id) return; // выбрали другой кадр
    let reposition = false;
    if (!state.forward || recenter) {
      const f = forwardOf(frame);
      if (f || !state.forward) {
        state.forward = f || state.forward;
        reposition = true;
      }
    }
    state.frameData = frame;
    state.cloudBuf = buf;
    disposeGroup(groups.cloud);
    disposeGroup(groups.overlay);
    groups.cloud = cloudObject(buf, frame);
    groups.overlay = overlayObject(frame);
    scene.add(groups.cloud, groups.overlay);
    if (reposition) applyCam(state.cam);
    if (!groups.grid && frame.corridor) {
      groups.grid = gridObject(frame);
      scene.add(groups.grid);
    }
    $("view-cap").innerHTML =
      `<b>${esc(state.job.bag.name)}</b> · кадр ${k} · ${fmt(frame.t, 2)} с`;
    renderNow(frame);
    renderRecStatus(frame);
  } catch (e) {
    if (seq === state.seq) showError(e.message);
  }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  const w = view.clientWidth;
  const h = view.clientHeight;
  const v = new THREE.Vector3();
  for (const l of labels) {
    v.copy(l.pos).project(camera);
    const visible = v.z < 1 && Math.abs(v.x) <= 1.05 && Math.abs(v.y) <= 1.05;
    l.el.style.display = visible ? "" : "none";
    if (visible) {
      l.el.style.left = `${((v.x + 1) / 2) * w}px`;
      l.el.style.top = `${((1 - v.y) / 2) * h}px`;
    }
  }
}

// Статус-пилюля режима записи: крупно поверх вьюпорта.
function renderRecStatus(frame) {
  const el = $("rec-status");
  if (!frame) {
    el.className = "rec-status hidden";
    return;
  }
  const st = statusOf(frame);
  el.className = `rec-status ${st.cls}`;
  el.textContent = st.text;
}

// ------------------------------------------------------------------ проигрывание

const PLAY_ICON = '<path d="M4 2.5v11l9-5.5z" fill="currentColor" />';
const PAUSE_ICON =
  '<rect x="3.5" y="2.5" width="3" height="11" fill="currentColor" /><rect x="9.5" y="2.5" width="3" height="11" fill="currentColor" />';

function stopPlay() {
  state.playing = false;
  $("play-icon").innerHTML = PLAY_ICON;
}

// Темп — по часам: кадр = старт + прошедшее время × скорость / шаг записи.
// Если кадр грузится дольше шага, промежуточные пропускаются, а темп сохраняется.
async function playLoop() {
  const n = state.series.t.length;
  let startFrame = state.frame + 1 >= n ? 0 : state.frame;
  let t0 = performance.now();
  let speed = state.speed;
  while (state.playing) {
    if (speed !== state.speed) {
      speed = state.speed;
      startFrame = state.frame;
      t0 = performance.now();
    }
    const dt = frameDt();
    const target = startFrame + Math.floor(((performance.now() - t0) / 1000) * (speed / dt)) + 1;
    const len = state.series.t.length;
    if (target >= len) {
      if (state.job.state === "done") {
        stopPlay();
        break;
      }
      await new Promise((r) => setTimeout(r, 100));
      continue;
    }
    if (target !== state.frame) await showFrame(target);
    else await new Promise((r) => setTimeout(r, 10));
  }
}

function togglePlay() {
  if (!state.series || !state.series.t.length) return;
  if (state.playing) {
    stopPlay();
    return;
  }
  state.playing = true;
  $("play-icon").innerHTML = PAUSE_ICON;
  playLoop();
}

$("play").addEventListener("click", (e) => {
  e.currentTarget.blur(); // иначе Пробел нажмёт кнопку второй раз
  togglePlay();
});
$("speeds").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-speed]");
  if (!b) return;
  state.speed = Number(b.dataset.speed);
  for (const x of $("speeds").querySelectorAll("button")) x.classList.toggle("on", x === b);
});

$("events").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-frame]");
  if (!tr) return;
  stopPlay();
  showFrame(Number(tr.dataset.frame));
});

$("zoom-in").addEventListener("click", () => zoom(1 / 1.35));
$("zoom-out").addEventListener("click", () => zoom(1.35));
$("reset-view").addEventListener("click", () => applyCam(state.cam));
$("campresets").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-cam]");
  if (b) setCam(b.dataset.cam);
});
$("color-mode").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-color]");
  if (!b || b.dataset.color === state.color) return;
  state.color = b.dataset.color;
  syncColorUi();
  syncUrl();
  if (state.cloudBuf && state.frameData) rebuildCloud();
});

function syncColorUi() {
  for (const b of $("color-mode").querySelectorAll("button[data-color]"))
    b.classList.toggle("on", b.dataset.color === state.color);
  $("ramp-label").textContent =
    state.color === "intensity" ? "интенсивность" : "высота над полотном";
}
$("fit-view").addEventListener("click", fitView);
$("dl-mcap").addEventListener("click", onMcap);
$("run-form").addEventListener("submit", onRun);
$("bag-file").addEventListener("change", () => {
  const f = $("bag-file").files[0];
  setFilePicked(f);
  if (f) $("bag-select").value = "";
});
$("bag-select").addEventListener("change", () => {
  $("bag-file").value = "";
  setFilePicked(null);
});
window.addEventListener("keydown", (e) => {
  const tag = e.target.tagName;
  if (!state.series || tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  const step = e.shiftKey ? 10 : 1;
  if (e.key === "ArrowRight") {
    e.preventDefault();
    stopPlay();
    showFrame(state.frame + step);
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    stopPlay();
    showFrame(state.frame - step);
  } else if (e.key === "Home") {
    stopPlay();
    showFrame(0);
  } else if (e.key === "End") {
    stopPlay();
    showFrame(state.series.t.length - 1);
  } else if (e.key === " ") {
    e.preventDefault();
    togglePlay();
  } else if (e.key >= "1" && e.key <= "3") {
    setCam(CAMS[Number(e.key) - 1]);
  }
});

// ------------------------------------------------------------------ старт

document.body.classList.toggle("rec", REC);
for (const b of $("campresets").querySelectorAll("button[data-cam]"))
  b.classList.toggle("on", b.dataset.cam === state.cam);
syncColorUi();
resize();
animate();
renderNow(null);
renderEmptyLeft();
renderDownloads(null);
drawTracks();
loadBags().catch((e) => showError(e.message));
loadJobs();
const m = /job=([\w-]+)/.exec(location.hash);
if (m) openJob(m[1]);
