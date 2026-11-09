
(() => {
  const MAX_PTS = 96;
  const FIELD_RES = 256, FIELD_SIZE = 32;     // world window around the boat, texel = 0.125
  const canvas = document.getElementById('c');
  const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
  if (!gl) { document.getElementById('err').style.display = 'grid'; return; }

  const src = id => document.getElementById(id).textContent;
  const fragSrc = id => '#version 300 es\nprecision highp float;\nprecision highp int;\n' + src('common') + src(id);
  function compile(type, s) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, s.trim());
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  }
  function program(fs, names) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, src('vs')));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, 'aPos');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const U = {};
    for (const n of names) U[n] = gl.getUniformLocation(p, n);
    return { p, U };
  }
  const fieldProg = program(fragSrc('fs-field'), ['uOrg', 'uSize', 'uRes', 'uTime', 'uPts', 'uAux', 'uCount']);
  const mainProg = program(fragSrc('fs'), ['uRes', 'uTime', 'uCamPos', 'uCamR', 'uCamU', 'uCamF', 'uTanH',
    'uBoatPos', 'uBoatRot', 'uFinPos', 'uUpGlow', 'uField', 'uFieldOrg', 'uFieldSize', 'uSmoke']);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  // field render target: half float when renderable (precise cut lines), else RGBA8
  const fieldTex = gl.createTexture();
  const fbo = gl.createFramebuffer();
  function makeField(useHalf) {
    gl.bindTexture(gl.TEXTURE_2D, fieldTex);
    if (useHalf) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, FIELD_RES, FIELD_RES, 0, gl.RGBA, gl.HALF_FLOAT, null);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, FIELD_RES, FIELD_RES, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fieldTex, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return ok;
  }
  const canHalf = !!(gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float'));
  if (!(canHalf && makeField(true))) makeField(false);

  // ---------- layout (adaptive resolution: the scene shader is heavy) ----------
  let W = 0, H = 0, scale = 1, maxScale = 1;
  function applySize() {
    canvas.width = Math.max(1, Math.round(W * scale));
    canvas.height = Math.max(1, Math.round(H * scale));
  }
  function resize() {
    W = innerWidth; H = innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    maxScale = Math.min(dpr, Math.sqrt(2.0e6 / (W * H)));
    scale = Math.min(scale, maxScale) || maxScale;
    applySize();
  }
  addEventListener('resize', resize);
  scale = 10; resize();

  // ---------- small vector helpers ----------
  const norm = v => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const smooth = x => { x = Math.min(Math.max(x, 0), 1); return x * x * (3 - 2 * x); };
  const wrapA = a => Math.atan2(Math.sin(a), Math.cos(a));
  function mul3(A, B) {
    const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    return C;
  }

  // ---------- camera: fixed orientation, follows the boat (steering by pointing stays intuitive) ----------
  const TANH = Math.tan(31 * Math.PI / 180);
  const CAM_OFF = [0, 5.0, -8.2], LOOK_AHEAD = 7.0;   // high and far: lots of empty world
  // the camera swings slowly round behind the boat's heading (a few seconds' lag keeps pointing intuitive)
  const cam = { fx: 0, fz: 0, yaw: 0, pull: 1, pos: [0, 0, 0], R: [1, 0, 0], U: [0, 1, 0], F: [0, 0, 1] };
  function updateCamera(dt = 0) {
    const sx = Math.sin(cam.yaw), cz = Math.cos(cam.yaw);
    const want = [cam.fx + sx * CAM_OFF[2], CAM_OFF[1], cam.fz + cz * CAM_OFF[2]];
    // never sit inside a berg or with glass between camera and boat: walk the boat-to-camera line
    let frac = 1;
    if (typeof bergDist === 'function') {
      for (let i = 1; i <= 24; i++) {
        const u = i / 24;
        const x = boat.x + (want[0] - boat.x) * u, y = 1.2 + (want[1] - 1.2) * u, z = boat.z + (want[2] - boat.z) * u;
        if (bergDist(x, y, z) < 0.9) { frac = Math.max(0.22, (i - 1.5) / 24); break; }
      }
    }
    cam.pull += (frac - cam.pull) * (1 - Math.exp(-dt * (frac < cam.pull ? 6 : 1.2)));   // pull in briskly, ease back out
    const pl = cam.pull;
    cam.pos = [boat.x + (want[0] - boat.x) * pl, 1.2 + (want[1] - 1.2) * pl + (1 - pl) * 1.6, boat.z + (want[2] - boat.z) * pl];
    cam.F = norm([cam.fx + sx * LOOK_AHEAD - cam.pos[0], -cam.pos[1], cam.fz + cz * LOOK_AHEAD - cam.pos[2]]);
    cam.R = norm(cross([0, 1, 0], cam.F));
    cam.U = cross(cam.F, cam.R);
  }
  function rayDir(px, py) {
    const x = (2 * px - W) / H * TANH, y = (H - 2 * py) / H * TANH;
    return norm([cam.R[0] * x + cam.U[0] * y + cam.F[0], cam.R[1] * x + cam.U[1] * y + cam.F[1], cam.R[2] * x + cam.U[2] * y + cam.F[2]]);
  }
  function onPlane(px, py) {
    const d = rayDir(px, py);
    const dy = Math.min(d[1], -0.04);
    const t = Math.min(-cam.pos[1] / dy, 45);
    return [cam.pos[0] + d[0] * t, cam.pos[2] + d[2] * t];
  }
  function screenX01(x, y, z) {   // for stereo panning
    const r = [x - cam.pos[0], y - cam.pos[1], z - cam.pos[2]];
    const ndc = dot3(r, cam.R) / (Math.max(dot3(r, cam.F), 0.1) * TANH);
    return Math.min(Math.max(0.5 + ndc * H / (2 * W), 0), 1);
  }

  // ---------- sound (Web Audio, fully synthesized) ----------
  const Sound = (() => {
    let ctx = null, master, on = false;
    let rumbleG, hissG, sizzleG, sizzleF, windG, windF, humG, humA, humB, white, brown;
    let lastCrack = 0, lastCrunch = 0, lastGloop = 0, susTimer = 0;
    function noise(kind) {
      const len = ctx.sampleRate * 2, b = ctx.createBuffer(1, len, ctx.sampleRate), d = b.getChannelData(0);
      let last = 0;
      for (let i = 0; i < len; i++) {
        const w = Math.random() * 2 - 1;
        if (kind === 'brown') { last = (last + 0.02 * w) / 1.02; d[i] = last * 3.5; } else d[i] = w;
      }
      if (kind === 'brown') { const e = d[len - 1] - d[0]; for (let i = 0; i < len; i++) d[i] -= e * i / (len - 1); }
      return b;
    }
    function loop(buf) { const s = ctx.createBufferSource(); s.buffer = buf; s.loop = true; s.start(0, Math.random() * buf.duration); return s; }
    function filt(type, f, q = 0.7) { const b = ctx.createBiquadFilter(); b.type = type; b.frequency.value = f; b.Q.value = q; return b; }
    function out(pan) { const p = ctx.createStereoPanner(); p.pan.value = Math.max(-1, Math.min(1, pan)); p.connect(master); return p; }
    function env(g, t, a, peak, d) {
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(peak, t + a);
      g.gain.exponentialRampToValueAtTime(0.0001, t + a + d);
    }
    function gainNode(v = 0) { const g = ctx.createGain(); g.gain.value = v; g.connect(master); return g; }
    function init() {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      master = ctx.createGain(); master.gain.value = 0;
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -16; comp.ratio.value = 4;
      master.connect(comp).connect(ctx.destination);
      white = noise('white'); brown = noise('brown');
      // magma rumble, swelling with how much of the wake is open
      rumbleG = gainNode();
      loop(brown).connect(filt('lowpass', 140)).connect(rumbleG);
      // cooling hiss while the wake glues back
      hissG = gainNode();
      loop(white).connect(filt('highpass', 3800)).connect(filt('lowpass', 9000)).connect(hissG);
      // the fin slicing through magma: a band of sizzle that brightens with speed
      sizzleG = gainNode(); sizzleF = filt('bandpass', 1800, 0.9);
      loop(white).connect(sizzleF).connect(sizzleG);
      // lonely wind over the crust, slowly gusting; stronger with speed
      windG = gainNode(); windF = filt('bandpass', 650, 0.5);
      loop(brown).connect(windF).connect(windG);
      const gust = ctx.createOscillator(); gust.frequency.value = 0.07;
      const gustG = ctx.createGain(); gustG.gain.value = 260; gust.connect(gustG).connect(windF.frequency); gust.start();
      // levitation orbs: a low, soft drone a fifth apart with a slow beat
      humG = gainNode();
      const humF = filt('lowpass', 380, 0.8); humF.connect(humG);
      humA = ctx.createOscillator(); humA.type = 'sine'; humA.frequency.value = 49;
      humB = ctx.createOscillator(); humB.type = 'sine'; humB.frequency.value = 73.6;
      const gA = ctx.createGain(); gA.gain.value = 0.7; const gB = ctx.createGain(); gB.gain.value = 0.35;
      humA.connect(gA).connect(humF); humB.connect(gB).connect(humF);
      const humC = ctx.createOscillator(); humC.type = 'sine'; humC.frequency.value = 196;   // audible on laptop speakers
      const gC = ctx.createGain(); gC.gain.value = 0.16; humC.connect(gC).connect(humF); humC.start();
      const trem = ctx.createOscillator(); trem.frequency.value = 3.2;
      const tremG = ctx.createGain(); tremG.gain.value = 0.18; trem.connect(tremG).connect(gB.gain);
      humA.start(); humB.start(); trem.start();
    }
    // sharp stone crackle: a cluster of tiny band-passed noise ticks
    function crackle(x, strength) {
      if (!on) return;
      const t0 = ctx.currentTime;
      if (t0 - lastCrack < 0.035) return;
      lastCrack = t0;
      const o = out((x - 0.5) * 1.4);
      const count = 2 + Math.floor(strength * 6 * Math.random());
      for (let i = 0; i < count; i++) {
        const t = t0 + (i ? Math.random() * 0.14 : 0);
        const s = ctx.createBufferSource(); s.buffer = white;
        const g = ctx.createGain();
        const bp = filt('bandpass', 1400 + Math.random() * 5200, 2 + Math.random() * 7);
        s.connect(bp).connect(g).connect(o);
        const dur = 0.004 + Math.random() * 0.03;
        env(g, t, 0.001, (0.12 + 0.4 * Math.random()) * strength, dur);
        s.start(t, Math.random() * 1.5, dur + 0.02);
      }
    }
    // deep crunch for a bigger break
    function crunch(x, strength) {
      if (!on) return;
      if (ctx.currentTime - lastCrunch < 0.09) return crackle(x, strength);
      lastCrunch = ctx.currentTime;
      const t = ctx.currentTime, o = out((x - 0.5) * 1.2);
      const s = ctx.createBufferSource(); s.buffer = white;
      const g = ctx.createGain();
      s.connect(filt('lowpass', 520, 1.2)).connect(g).connect(o);
      env(g, t, 0.003, 0.55 * strength, 0.12);
      s.start(t, Math.random() * 1.5, 0.2);
      const th = ctx.createOscillator(), tg = ctx.createGain();
      th.frequency.setValueAtTime(95, t); th.frequency.exponentialRampToValueAtTime(38, t + 0.14);
      th.connect(tg).connect(o); env(tg, t, 0.004, 0.35 * strength, 0.16);
      th.start(t); th.stop(t + 0.25);
      crackle(x, strength);
    }
    // steam chuff from the funnel
    function chuff(x, strength) {
      if (!on) return;
      const t = ctx.currentTime, o = out((x - 0.5) * 1.2);
      const s = ctx.createBufferSource(); s.buffer = white;
      const g = ctx.createGain(), lp = filt('lowpass', 1100, 0.9);
      lp.frequency.setValueAtTime(1300, t); lp.frequency.exponentialRampToValueAtTime(350, t + 0.2);
      s.connect(filt('highpass', 120)).connect(lp).connect(g).connect(o);
      env(g, t, 0.012, 0.09 * strength, 0.22);
      s.start(t, Math.random() * 1.5, 0.3);
    }
    // timber creak: a sawtooth through a narrow band, stick-slip chopped
    function creak(x) {
      if (!on) return;
      const t = ctx.currentTime, o = out((x - 0.5) * 1.2);
      const f = 70 + Math.random() * 60, dur = 0.25 + Math.random() * 0.35;
      const osc = ctx.createOscillator(); osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(f, t);
      osc.frequency.linearRampToValueAtTime(f * (1.1 + Math.random() * 0.25), t + dur * 0.5);
      osc.frequency.linearRampToValueAtTime(f * 0.92, t + dur);
      const bp = filt('bandpass', 500 + Math.random() * 500, 7);
      const chop = ctx.createGain(); chop.gain.value = 0.5;
      const slip = ctx.createOscillator(); slip.type = 'square'; slip.frequency.value = 18 + Math.random() * 20;
      const slipG = ctx.createGain(); slipG.gain.value = 0.5; slip.connect(slipG).connect(chop.gain);
      const g = ctx.createGain();
      osc.connect(bp).connect(chop).connect(g).connect(o);
      env(g, t, 0.06, 0.06, dur);
      osc.start(t); slip.start(t); osc.stop(t + dur + 0.1); slip.stop(t + dur + 0.1);
    }
    function bubble(x) {
      if (!on) return;
      const t = ctx.currentTime, o = out((x - 0.5) * 1.4);
      const f0 = 120 + Math.random() * 240;
      const osc = ctx.createOscillator(), g = ctx.createGain();
      osc.frequency.setValueAtTime(f0, t);
      osc.frequency.exponentialRampToValueAtTime(f0 * 0.45, t + 0.1);
      osc.connect(filt('lowpass', 900)).connect(g).connect(o);
      env(g, t, 0.006, 0.07 + Math.random() * 0.07, 0.11);
      osc.start(t); osc.stop(t + 0.2);
    }
    function gloop(x) {
      if (!on) return;
      if (ctx.currentTime - lastGloop < 0.08) return;
      lastGloop = ctx.currentTime;
      const t = ctx.currentTime + Math.random() * 0.05, o = out((x - 0.5) * 1.4);
      const osc = ctx.createOscillator(), g = ctx.createGain(), lp = filt('lowpass', 1400, 3);
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(210 + Math.random() * 60, t);
      osc.frequency.exponentialRampToValueAtTime(80, t + 0.2);
      lp.frequency.setValueAtTime(1400, t); lp.frequency.exponentialRampToValueAtTime(260, t + 0.22);
      osc.connect(lp).connect(g).connect(o);
      env(g, t, 0.012, 0.08, 0.22);
      osc.start(t); osc.stop(t + 0.3);
    }
    function update(m, dt) {
      if (!on) return;
      const t = ctx.currentTime, sp = Math.min(1, m.speed / 6);
      rumbleG.gain.setTargetAtTime(Math.min(1, m.open * 0.5) * 0.5, t, 0.25);
      hissG.gain.setTargetAtTime(Math.min(1, m.healing * 0.08) * 0.05, t, 0.15);
      sizzleG.gain.setTargetAtTime(0.012 + sp * 0.07, t, 0.1);
      sizzleF.frequency.setTargetAtTime(1500 + sp * 2600, t, 0.15);
      windG.gain.setTargetAtTime(0.09 + sp * sp * 0.25, t, 0.3);
      humG.gain.setTargetAtTime(0.1 + sp * 0.04, t, 0.2);
      humA.frequency.setTargetAtTime(49 + sp * 6 + m.turn * 3, t, 0.3);
      humB.frequency.setTargetAtTime(73.6 + sp * 9 + m.turn * 4.5, t, 0.3);
      if (Math.random() < Math.min(m.open, 6) * 0.6 * dt) bubble(m.x);
    }
    function setOn(v) {
      if (!ctx) init();
      on = v;
      const t = ctx.currentTime;
      clearTimeout(susTimer);
      if (on) ctx.resume();
      else {
        for (const g of [rumbleG, hissG, sizzleG, windG, humG]) { g.gain.cancelScheduledValues(t); g.gain.setTargetAtTime(0, t, 0.05); }
        susTimer = setTimeout(() => { if (!on) ctx.suspend(); }, 400);
      }
      master.gain.setTargetAtTime(on ? 0.9 : 0, t, 0.05);
    }
    const safe = fn => (...a) => { try { return fn(...a); } catch (e) { on = false; console.warn('sound disabled:', e); } };
    return { crackle: safe(crackle), crunch: safe(crunch), gloop: safe(gloop), update: safe(update),
             chuff: safe(chuff), creak: safe(creak),
             setOn, get on() { return on; } };
  })();

  const sndBtn = document.getElementById('snd');
  function toggleSound() {
    Sound.setOn(!Sound.on);
    sndBtn.textContent = Sound.on ? 'Sound on' : 'Sound off · click to enable';
    sndBtn.classList.toggle('on', Sound.on);
    sndBtn.setAttribute('aria-pressed', String(Sound.on));
  }
  sndBtn.addEventListener('click', toggleSound);
  addEventListener('keydown', e => {
    if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'm' || e.key === 'M') toggleSound();
  });

  // ---------- pointer ----------
  const ptr = { x: 0, y: 0, active: false };
  const hint = document.getElementById('hint');
  function onMove(e) {
    ptr.x = e.clientX; ptr.y = e.clientY; ptr.active = true;
    if (!hint.classList.contains('gone')) setTimeout(() => hint.classList.add('gone'), 2500);
  }
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerdown', onMove);
  canvas.addEventListener('pointerup', e => { if (e.pointerType === 'touch') ptr.active = false; });
  canvas.addEventListener('pointerleave', () => { ptr.active = false; });
  canvas.addEventListener('pointercancel', () => { ptr.active = false; });

  // ---------- obsidian bergs: exact mirror of the shader's layout, for collision ----------
  function pcg2d(x, y) {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0; y = (Math.imul(y, 1664525) + 1013904223) >>> 0;
    x = (x + Math.imul(y, 1664525)) >>> 0; y = (y + Math.imul(x, 1664525)) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0; y = (y ^ (y >>> 16)) >>> 0;
    x = (x + Math.imul(y, 1664525)) >>> 0; y = (y + Math.imul(x, 1664525)) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0; y = (y ^ (y >>> 16)) >>> 0;
    return [x / 4294967295, y / 4294967295];
  }
  const hash2 = (x, y) => pcg2d(Math.floor(x + 0.5) | 0, Math.floor(y + 0.5) | 0);
  const BC = 30;
  const BA = [[1, 0], [0.5254, 0.8508], [-0.1439, 0.9896], [-0.9215, 0.3883], [-0.9398, -0.3418], [-0.1538, -0.9881], [0.5917, -0.8061]];
  const BE = [[0.9968, 0.0799], [0.9838, 0.1790], [0.9928, 0.1197], [0.9759, 0.2182], [0.9988, -0.0500], [0.9888, 0.1494], [0.9950, 0.0998]];
  const bergCache = new Map();
  function makeBerg(cx, cz) {   // a cluster of three leaning shards, exactly as in the shader
    const key = cx + ',' + cz;
    let b = bergCache.get(key);
    if (b) return b;
    const h = hash2(cx, cz), h2 = hash2(cx + 57, cz + 113), h3 = hash2(cx + 211, cz + 17);
    b = { on: h[0] < 0.22 && !(Math.abs(cx + 0.5) < 1 && Math.abs(cz + 0.5) < 1), shards: [] };
    const c0 = [(cx + 0.5 + (h2[0] - 0.5) * 0.45) * BC, (cz + 0.5 + (h2[1] - 0.5) * 0.45) * BC];
    const big = h3[1] > 0.86 ? 1.7 : 1;
    for (let i = 0; i < 3; i++) {
      const r = hash2(cx * 4 + i, cz * 4 + 1), r2 = hash2(cx * 4 + i, cz * 4 + 2);
      const sc = (i === 0 ? 1 : 0.45 + 0.4 * r[0]) * big;
      const R = (0.9 + 1.1 * h3[0]) * sc, yaw = r[1] * 6.2832, a = 6.2832 * r2[0];
      const off = i === 0 ? 0 : (1.4 + 1.2 * r2[1]) * Math.min(big, 1.3);
      const H = (3.5 + 6.0 * h3[0]) * sc * (0.7 + 0.6 * r2[1]);
      const la = a + 0.6 * (r[0] - 0.5), lm = Math.min(0.12 + 0.25 * r2[1], 3.5 / H);
      const ca = 6.2832 * hash2(cx * 4 + i, cz * 4 + 3)[0], ct = 0.55 + 0.4 * r[0];
      const offs = [];
      for (let k = 0; k < 7; k++) offs.push(R * (0.7 + 0.6 * hash2(cx * 32 + i * 7 + k, cz * 32 + 9)[0]));
      b.shards.push({ c: [c0[0] + Math.cos(a) * off, c0[1] + Math.sin(a) * off], cs: [Math.cos(yaw), Math.sin(yaw)],
        lean: [Math.cos(la) * lm, Math.sin(la) * lm], cap: [Math.sin(ct) * Math.cos(ca), Math.cos(ct), Math.sin(ct) * Math.sin(ca)], H, offs });
    }
    if (bergCache.size > 400) bergCache.clear();
    bergCache.set(key, b);
    return b;
  }
  function shardDist(sh, x, y, z) {
    const qx = x - sh.c[0] - sh.lean[0] * y, qz = z - sh.c[1] - sh.lean[1] * y, cs = sh.cs;
    let d = sh.cap[0] * qx + sh.cap[1] * y + sh.cap[2] * qz - sh.H * sh.cap[1];
    for (let k = 0; k < 7; k++) {
      const ax = BA[k][0] * cs[0] - BA[k][1] * cs[1], az = BA[k][0] * cs[1] + BA[k][1] * cs[0];
      d = Math.max(d, BE[k][0] * ax * qx + BE[k][1] * y + BE[k][0] * az * qz - sh.offs[k]);
    }
    return d;
  }
  function bergDist(x, y, z) {
    const cx = Math.floor(x / BC), cz = Math.floor(z / BC);
    let d = 1e9;
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
      const b = makeBerg(cx + i, cz + j);
      if (b.on) for (const sh of b.shards) d = Math.min(d, shardDist(sh, x, y, z));
    }
    return d;
  }
  // keep bow, middle and stern clear of the glass.
  //  - fresh contact: commit to sliding one way along the face; pointer steering pauses briefly (armed once per
  //    contact, so steering always comes back)
  //  - a notch (probes disagree, or the contact normal flips between frames) or no progress for 0.6 s:
  //    back away from the glass on a fixed escape heading for a moment, then try the other side
  const probeDist = (x, z, high) => high ? Math.min(bergDist(x, 0.4, z), bergDist(x, 1.4, z)) : bergDist(x, 0.4, z);
  function collide(dt) {
    let hit = 0, nx = 0, nz = 0;
    const fx = Math.sin(boat.yaw), fz = Math.cos(boat.yaw);
    for (const [along, clear, high] of [[2.3, 0.3, true], [1.2, 0.75, false], [0.0, 1.0, false], [-0.8, 0.9, false], [-1.6, 0.6, false]]) {
      const px = boat.x + fx * along, pz = boat.z + fz * along;
      const d = probeDist(px, pz, high);
      if (d >= clear) continue;
      const e = 0.05;
      const gx = probeDist(px + e, pz, high) - probeDist(px - e, pz, high);
      const gz = probeDist(px, pz + e, high) - probeDist(px, pz - e, high);
      const gl = Math.hypot(gx, gz) || 1;
      boat.x += gx / gl * (clear - d); boat.z += gz / gl * (clear - d);
      nx += gx / gl; nz += gz / gl; hit++;
    }
    if (boat.escapeT > 0) boat.escapeT -= dt;
    if (!hit) {
      boat.inContact = false; boat.freeT += dt;
      if (boat.freeT < 0.3) boat.cT += dt;   // brief gaps between touches still count toward the no-progress window
      if (boat.freeT > 0.3) boat.slideSign = 0;
      return false;
    }
    const nl = Math.hypot(nx, nz);
    const ux = nl > 1e-3 ? nx / nl : -fx, uz = nl > 1e-3 ? nz / nl : -fz;
    if (!boat.inContact && boat.freeT > 0.3) {          // fresh contact
      boat.contactT = 0.9; boat.cx0 = boat.x; boat.cz0 = boat.z; boat.cT = 0;
    }
    boat.cT += dt;
    const flipped = boat.inContact && ux * boat.lnx + uz * boat.lnz < 0.3;
    const notch = (hit > 1 && nl < 0.6 * hit) || flipped;
    let noProgress = false;
    if (boat.cT > 0.6) {   // sliding window: judge only the last 0.6 s of contact
      noProgress = Math.hypot(boat.x - boat.cx0, boat.z - boat.cz0) < 0.4;
      boat.cx0 = boat.x; boat.cz0 = boat.z; boat.cT = 0;
    }
    if ((notch || noProgress) && boat.escapeT <= 0) {
      // back off: away from the glass if the normal is clear, otherwise the way we came
      boat.escapeDir = nl > 0.6 * hit ? Math.atan2(ux, uz) : Math.atan2(-fx, -fz);
      boat.escapeT = 0.7;
      boat.slideSign = boat.slideSign ? -boat.slideSign : 1;
      boat.cx0 = boat.x; boat.cz0 = boat.z; boat.cT = 0;
    }
    // steer for the boat only while escaping or actually heading into the glass; otherwise the pointer is free
    let dir = null;
    if (boat.escapeT > 0) dir = boat.escapeDir;
    else if (fx * ux + fz * uz < 0.1) {
      if (!boat.slideSign) boat.slideSign = (fx * -uz + fz * ux) >= 0 ? 1 : -1;
      const tx = -uz * boat.slideSign, tz = ux * boat.slideSign;
      dir = Math.atan2(tx + ux * 0.35, tz + uz * 0.35);
    }
    if (dir !== null) {
      boat.yaw = wrapA(boat.yaw + wrapA(dir - boat.yaw) * Math.min(1, dt * 4));
      boat.yawRate *= Math.exp(-dt * 6);
    }
    boat.v = boat.contactT > 0 || boat.escapeT > 0 ? Math.max(boat.v * Math.exp(-dt * 0.8), 1.2) : boat.v * Math.exp(-dt * 1.5);
    boat.inContact = true; boat.freeT = 0; boat.lnx = ux; boat.lnz = uz;
    return true;
  }

  // ---------- boat ----------
  const MAXV = 6.2;
  const boat = { x: 0, z: 0, yaw: 0, v: 0, yawRate: 0, roll: 0, pitch: 0, y: 0.78, scrape: 0, contactT: 0, slideSign: 0,
                 escapeT: 0, escapeDir: 0, inContact: false, freeT: 1, cT: 0, cx0: 0, cz0: 0, lnx: 0, lnz: 0 };
  window.__magmaBoat = { boat, ptr, cam, bergDist: (x, z) => bergDist(x, 0.3, z) };   // for poking at it from the console
  const FIN_LOCAL = [0, -0.78, -1.21];   // where the swept keel crosses the crust (boat frame, world units)
  let boatRotCols = new Float32Array(9), boatRows = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  function updateBoat(dt, t) {
    let wantRate = 0, throttle = 0, tgtDist = Infinity;
    if (ptr.active) {
      const T = onPlane(ptr.x, ptr.y);
      const dx = T[0] - boat.x, dz = T[1] - boat.z, dist = Math.hypot(dx, dz);
      tgtDist = dist;
      if (dist > 0.9) {
        const err = wrapA(Math.atan2(dx, dz) - boat.yaw);
        const turnMax = 1.0 + 1.2 * Math.min(1, boat.v / 3);
        wantRate = Math.max(-turnMax, Math.min(turnMax, err * 2.4));
        throttle = Math.min(1, (dist - 0.9) / 12) * Math.pow(Math.max(0, Math.cos(err)), 1.5);
      }
    }
    if (boat.contactT > 0) boat.contactT -= dt;
    if (boat.contactT > 0 || boat.escapeT > 0) {   // sliding / backing off the glass: hold course, keep a little way on
      wantRate = 0; if (tgtDist > 2.5) throttle = Math.max(throttle, 0.3);   // no forced way-on when the pointer is right here
    }
    boat.yawRate += (wantRate - boat.yawRate) * (1 - Math.exp(-dt * 5));
    boat.yaw = wrapA(boat.yaw + boat.yawRate * dt);
    const vT = MAXV * throttle, v0 = boat.v;
    boat.v += (vT - boat.v) * (1 - Math.exp(-dt * (vT > boat.v ? 1.1 : 0.8)));
    boat.x += Math.sin(boat.yaw) * boat.v * dt;
    boat.z += Math.cos(boat.yaw) * boat.v * dt;
    boat.scrape = collide(dt) ? 1 : 0;
    const acc = (boat.v - v0) / Math.max(dt, 1e-3);
    const rollT = Math.max(-0.35, Math.min(0.35, -boat.yawRate * (0.12 + boat.v * 0.06)));
    boat.roll += (rollT - boat.roll) * (1 - Math.exp(-dt * 4));
    const pitchT = Math.max(-0.06, Math.min(0.09, acc * 0.03)) + 0.012 * Math.sin(t * 1.3);
    boat.pitch += (pitchT - boat.pitch) * (1 - Math.exp(-dt * 3));
    boat.y = 0.78 + 0.035 * Math.sin(t * 1.6) + 0.018 * Math.sin(t * 2.7 + 1.0);
    const cy = Math.cos(boat.yaw), sy = Math.sin(boat.yaw), cp = Math.cos(boat.pitch), spp = Math.sin(boat.pitch);
    const cr = Math.cos(boat.roll), sr = Math.sin(boat.roll);
    boatRows = mul3(mul3([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], [[1, 0, 0], [0, cp, spp], [0, -spp, cp]]), [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]);
    const M = boatRows;
    boatRotCols = new Float32Array([M[0][0], M[1][0], M[2][0], M[0][1], M[1][1], M[2][1], M[0][2], M[1][2], M[2][2]]);
    return Math.abs(boat.yawRate);
  }
  function finWorld() {
    const M = boatRows, l = FIN_LOCAL;
    return [boat.x + M[0][0] * l[0] + M[0][1] * l[1] + M[0][2] * l[2],
            boat.z + M[2][0] * l[0] + M[2][1] * l[1] + M[2][2] * l[2]];
  }

  // ---------- wake: chronological points dropped by the fin ----------
  // Each point: the crust splits open behind the fin, widens, holds hot, then glues back without a trace.
  const OPEN = 0.8, HOLD = 0.9, HEAL = 1.8, LIFESPAN = HOLD + HEAL;
  const A0 = 0.05, AMAX = 0.32;
  const pts = [];   // {x, z, born, hx, hz, k, sealed}
  const lifeOf = age => age < HOLD ? 1 : 1 - smooth((age - HOLD) / HEAL);
  const amtOf = (age, k) => (A0 + (AMAX - A0) * k * smooth(age / OPEN)) * lifeOf(age);
  const reachOf = age => 0.75 + 0.85 * smooth(age / 1.2);

  // ---------- render loop ----------
  const ptsArr = new Float32Array(MAX_PTS * 4);
  const auxArr = new Float32Array(MAX_PTS * 4);
  const nowS = () => performance.now() / 1000;
  const t0 = nowS();
  let lastFrame = t0, perfAcc = 0, perfN = 0, perfT = t0, simT = 0;
  // funnel smoke: puffs leave the funnel top, rise, swell, drift downwind and thin out
  const SMOKE_N = 14, SMOKE_LIFE = 4.5, FUNNEL_L = [0, 1.42, 0.0];
  const puffs = [];
  const smokeArr = new Float32Array(SMOKE_N * 4);
  let nextPuff = 0, creakT = 2;
  cam.fx = boat.x; cam.fz = boat.z;

  function frame() {
    const now = nowS();
    const dt = Math.min(now - lastFrame, 0.1);
    lastFrame = now;
    const t = now - t0;
    simT += dt;   // ages run on the clamped sim clock, so a stall never fast-forwards the wake
    const sim = simT;

    // adaptive resolution: keep the frame time near 60 fps
    perfAcc += dt; perfN++;
    if (now - perfT > 1.0) {
      const avg = perfAcc / perfN;
      if (avg > 1 / 42 && scale > 0.45) { scale *= 0.85; applySize(); }
      else if (avg < 1 / 57 && scale < maxScale) { scale = Math.min(maxScale, scale * 1.08); applySize(); }
      perfAcc = 0; perfN = 0; perfT = now;
    }

    const turn = updateBoat(dt, t);
    const hx = Math.sin(boat.yaw), hz = Math.cos(boat.yaw);
    const k = 1 - Math.exp(-dt * 2.2);
    cam.fx += (boat.x + hx * boat.v * 0.35 - cam.fx) * k;
    cam.fz += (boat.z + hz * boat.v * 0.35 - cam.fz) * k;
    const swing = Math.max(Math.min(1, boat.v / 3), Math.min(1, Math.abs(boat.yawRate) / 0.6) * 0.8);
    if (swing > 0.1) cam.yaw = wrapA(cam.yaw + wrapA(boat.yaw - cam.yaw) * (1 - Math.exp(-dt * swing / 3.0)));
    updateCamera(dt);

    // drop wake points behind the fin; spacing grows with speed so the buffer spans a full lifespan
    const [fx, fz] = finWorld();
    const pan = screenX01(fx, 0, fz);
    const step = Math.max(0.16, boat.v * LIFESPAN / (MAX_PTS - 8));
    const lastP = pts[pts.length - 1];
    if (!lastP || Math.hypot(fx - lastP.x, fz - lastP.z) > step) {
      pts.push({ x: fx, z: fz, born: sim, hx, hz, k: 0.55 + 0.45 * Math.min(1, boat.v / 3), sealed: false });
      if (boat.v > 4.2 && Math.random() < 0.12) Sound.crunch(pan, 0.7);
      else if (Math.random() < 1.4 / (1 + boat.v * 0.25)) Sound.crackle(pan, Math.min(1, 0.35 + boat.v * 0.12));
    }
    while (pts.length && (sim - pts[0].born > LIFESPAN || pts.length > MAX_PTS - 1)) pts.shift();

    // smoke
    if (sim > nextPuff) {
      const M = boatRows, l = FUNNEL_L;
      puffs.push({ x: boat.x + M[0][0] * l[0] + M[0][1] * l[1] + M[0][2] * l[2],
                   y: boat.y + M[1][0] * l[0] + M[1][1] * l[1] + M[1][2] * l[2],
                   z: boat.z + M[2][0] * l[0] + M[2][1] * l[1] + M[2][2] * l[2], born: sim });
      if (puffs.length > SMOKE_N) puffs.shift();
      nextPuff = sim + 0.5 - 0.38 * Math.min(1, boat.v / 5);
      Sound.chuff(screenX01(boat.x, boat.y, boat.z), 0.5 + 0.5 * Math.min(1, boat.v / 5));
    }
    smokeArr.fill(-1);
    for (const p of puffs) {
      const a = (sim - p.born) / SMOKE_LIFE;
      p.y += (0.55 - 0.25 * a) * dt; p.x += 0.22 * dt; p.z += 0.05 * dt;
      p.a = a; p.d = (p.x - cam.pos[0]) ** 2 + (p.y - cam.pos[1]) ** 2 + (p.z - cam.pos[2]) ** 2;
    }
    // composite far to near so overlapping puffs layer correctly
    puffs.filter(p => p.a < 1).sort((u, v) => v.d - u.d).forEach((p, i) => smokeArr.set([p.x, p.y, p.z, p.a], i * 4));
    // the hull creaks now and then, more in hard turns and against the glass
    creakT -= dt * (0.15 + 1.6 * Math.min(1, turn) + 6 * boat.scrape);
    if (creakT < 0) { Sound.creak(screenX01(boat.x, boat.y, boat.z)); creakT = 1 + Math.random() * 2.5; }
    const m = { open: 0, healing: 0, speed: boat.v, turn: Math.min(1, turn), x: pan };
    let n = 0, near = 0;
    for (const p of pts) {
      const age = sim - p.born;
      const l = lifeOf(age), a = amtOf(age, p.k);
      m.open += a * 4;
      if (age > HOLD) m.healing += l;
      if (!p.sealed && age > HOLD + 0.1) { p.sealed = true; if (Math.random() < 0.06) Sound.gloop(screenX01(p.x, 0, p.z)); }
      const d2 = (p.x - boat.x) ** 2 + (p.z - boat.z) ** 2;
      near += a * l * Math.exp(-d2 / 5);
      ptsArr.set([p.x, p.z, l, a], n * 4);
      auxArr.set([reachOf(age), p.hx, p.hz, 0], n * 4);
      n++;
    }
    // live head at the fin: the slot is always open where the blade is
    ptsArr.set([fx, fz, 1, A0], n * 4);
    auxArr.set([reachOf(0), hx, hz, 0], n * 4);
    n++;
    m.open /= Math.max(1, pts.length * 0.25);
    Sound.update(m, dt);

    // pass 1: wake field (window snapped to texels so plate offsets never shimmer)
    const texel = FIELD_SIZE / FIELD_RES;
    const ox = Math.floor((cam.fx - FIELD_SIZE / 2) / texel) * texel;
    const oz = Math.floor((cam.fz + 4 - FIELD_SIZE / 2) / texel) * texel;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, FIELD_RES, FIELD_RES);
    gl.useProgram(fieldProg.p);
    gl.uniform2f(fieldProg.U.uOrg, ox, oz);
    gl.uniform1f(fieldProg.U.uSize, FIELD_SIZE);
    gl.uniform1f(fieldProg.U.uRes, FIELD_RES);
    gl.uniform1f(fieldProg.U.uTime, t);
    gl.uniform4fv(fieldProg.U.uPts, ptsArr);
    gl.uniform4fv(fieldProg.U.uAux, auxArr);
    gl.uniform1i(fieldProg.U.uCount, n);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // pass 2: scene
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(mainProg.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, fieldTex);
    const U = mainProg.U;
    gl.uniform1i(U.uField, 0);
    gl.uniform2f(U.uFieldOrg, ox, oz);
    gl.uniform1f(U.uFieldSize, FIELD_SIZE);
    gl.uniform2f(U.uRes, canvas.width, canvas.height);
    gl.uniform1f(U.uTime, t);
    gl.uniform3fv(U.uCamPos, cam.pos);
    gl.uniform3fv(U.uCamR, cam.R);
    gl.uniform3fv(U.uCamU, cam.U);
    gl.uniform3fv(U.uCamF, cam.F);
    gl.uniform1f(U.uTanH, TANH);
    gl.uniform3f(U.uBoatPos, boat.x, boat.y, boat.z);
    gl.uniformMatrix3fv(U.uBoatRot, false, boatRotCols);
    gl.uniform2f(U.uFinPos, fx, fz);
    gl.uniform1f(U.uUpGlow, Math.min(1, near * 2.5));
    gl.uniform4fv(U.uSmoke, smokeArr);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    schedule();
  }
  const schedule = () => document.hidden ? setTimeout(frame, 250) : requestAnimationFrame(frame);
  schedule();
})();
