#version 300 es
precision highp float;
precision highp int;

uvec2 pcg2d(uvec2 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * 1664525u; v.y += v.x * 1664525u;
  v ^= v >> 16u;
  v.x += v.y * 1664525u; v.y += v.x * 1664525u;
  v ^= v >> 16u;
  return v;
}
vec2 hash2(vec2 p) { return vec2(pcg2d(uvec2(ivec2(floor(p + 0.5))))) * (1.0 / 4294967295.0); }
float hash1(vec2 p) { return hash2(p).x; }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash1(i), b = hash1(i + vec2(1, 0)), c = hash1(i + vec2(0, 1)), d = hash1(i + vec2(1, 1));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbmN(vec2 p, int oct) {
  float s = 0.0, a = 0.5;
  mat2 r = mat2(0.8, -0.6, 0.6, 0.8);
  for (int i = 0; i < 6; i++) { if (i >= oct) break; s += a * vnoise(p); p = r * p * 2.03 + 11.7; a *= 0.5; }
  return s;
}
float fbm(vec2 p) { return fbmN(p, 5); }

uniform vec2  uRes;
uniform float uTime;
uniform vec3  uCamPos, uCamR, uCamU, uCamF;
uniform float uTanH;
uniform vec3  uBoatPos;
uniform mat3  uBoatRot;    // boat local (x right, y up, z forward) -> world
uniform vec2  uFinPos;     // where the fin enters the crust (world xz)
uniform float uUpGlow;     // how much open magma is near the boat (0..1)
uniform sampler2D uField;
uniform vec2  uFieldOrg;
uniform float uFieldSize;
uniform vec4  uSmoke[14];   // funnel smoke puffs: xyz world, w age 0..1 (<0 unused)
out vec4 fragColor;

const float S = 1.4;       // plates per world unit
const float SR = 0.4, DR = 2.5;
const float HM = 0.07;     // magma surface sits this far below the crust top (plate thickness)
const vec3 MOON = vec3(0.55, 0.58, 0.68);
const vec3 MOONDIR = vec3(-0.378, 0.1393, 0.9153);   // low moon, ahead and to the left
const vec3 AMB  = vec3(0.08, 0.044, 0.032);
const vec3 HAZE = vec3(0.30, 0.11, 0.045);
const vec3 AETHER = vec3(0.45, 1.0, 0.82);
const vec3 LANTC = vec3(1.0, 0.62, 0.26);
const vec3 LANT_L = vec3(0.0, 0.589, 1.02);   // bow lantern, boat local
const float BS = 1.35;                        // boat model scale (local units -> world)

// ---------- field ----------
vec4 fieldTex(vec2 w) {
  vec2 uv = (w - uFieldOrg) / uFieldSize;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return vec4(0.0, 0.5, 0.5, 1.0);
  return texture(uField, uv);
}
vec2 fSplit(vec4 f) { return (f.gb - 0.5) * 2.0 * SR; }
float fSd(vec4 f) { return (f.a - 0.5) * 2.0 * DR; }

// ---------- crust plates ----------
// Static warp gives plates jagged outlines. The low-frequency part is linearised (Jacobian) so a plate's
// outline moves rigidly with it; the small high-frequency raggedness is evaluated in place.
vec2 warpLow(vec2 u) { return 0.42 * vec2(fbmN(u * 0.57, 4), fbmN(u * 0.57 + 5.2, 4)) - 0.2; }
vec2 warpHi(vec2 u) {
  return 0.16 * vec2(fbmN(u * 2.0 + 2.0, 3), fbmN(u * 2.0 + 9.0, 3)) - 0.075
       + 0.07 * vec2(vnoise(u * 5.7), vnoise(u * 5.7 + 7.0)) - 0.035;
}

struct Q { float sd1, sd2, sdV1, cut1; vec2 o1, s1; };

// Each plate (or, where the fin's path crosses it, each half of a plate) is its Voronoi cell rigidly
// translated by the split field sampled at its site. Halves move to opposite sides of the path.
Q plates(vec2 X, vec2 P, mat2 M, bool live) {
  vec2 n = floor(X);
  vec2 site[25];
  for (int j = -2; j <= 2; j++)
  for (int i = -2; i <= 2; i++) {
    vec2 g = n + vec2(float(i), float(j));
    site[(j + 2) * 5 + (i + 2)] = g + 0.1 + 0.8 * hash2(g);
  }
  Q q;
  q.sd1 = 1e3; q.sd2 = 1e3; q.sdV1 = 1e3; q.cut1 = 0.0; q.o1 = vec2(0.0); q.s1 = vec2(0.0);
  for (int cj = -1; cj <= 1; cj++)
  for (int ci = -1; ci <= 1; ci++) {
    int k = (cj + 2) * 5 + (ci + 2);
    vec2 sk = site[k];
    vec2 v = live ? fSplit(fieldTex(sk / S)) : vec2(0.0);
    bool two = dot(v, v) > 1.6e-5;   // above the 8-bit encoding bias
    for (int s = 0; s < 2; s++) {
      if (s == 1 && !two) break;
      float sg = two ? (s == 0 ? 1.0 : -1.0) : 0.0;
      vec2 off = sg * v;                 // world
      vec2 Y = X - M * off;              // lattice point before the move
      float sdv = -1e3;
      for (int nj = -1; nj <= 1; nj++)
      for (int ni = -1; ni <= 1; ni++) {
        if (ni == 0 && nj == 0) continue;
        vec2 sj = site[k + nj * 5 + ni];
        sdv = max(sdv, dot(Y - 0.5 * (sk + sj), normalize(sj - sk)));
      }
      float sd = sdv, cut = 0.0;
      if (two) {   // keep only this half: the side of the fin's path it is moving toward
        // torn, ragged edge; noise in the half's rest coordinates so the edge rides rigidly with it
        vec2 Pr = P - off;
        float rag = 0.07 * (fbmN(Pr * 4.5, 2) - 0.37) + 0.025 * (vnoise(Pr * 17.0) - 0.5);
        float sdc = -sg * (fSd(fieldTex(Pr)) + rag) * S;
        cut = step(sdv, sdc);
        sd = max(sdv, sdc);
      }
      if (sd < q.sd1) { q.sd2 = q.sd1; q.sd1 = sd; q.o1 = off; q.s1 = sk; q.sdV1 = sdv; q.cut1 = cut; }
      else if (sd < q.sd2) { q.sd2 = sd; }
    }
  }
  return q;
}

float voronoiEdgeFree(vec2 x) {   // distance to nearest Voronoi border (IQ), full jitter
  vec2 n = floor(x), f = fract(x);
  vec2 mg, mr; float md = 8.0;
  for (int j = -1; j <= 1; j++)
  for (int i = -1; i <= 1; i++) {
    vec2 g = vec2(float(i), float(j));
    vec2 r = g + hash2(n + g) - f;
    float d = dot(r, r);
    if (d < md) { md = d; mr = r; mg = g; }
  }
  md = 8.0;
  for (int j = -2; j <= 2; j++)
  for (int i = -2; i <= 2; i++) {
    vec2 g = mg + vec2(float(i), float(j));
    vec2 r = g + hash2(n + g) - f;
    if (dot(mr - r, mr - r) > 1e-5) md = min(md, dot(0.5 * (mr + r), normalize(r - mr)));
  }
  return md;
}

// fine octaves fade with the pixel footprint so the mid-distance crust doesn't sparkle
float bumpH(vec2 y, vec2 k) { return vnoise(y * 3.1) * 0.6 + vnoise(y * 9.3 + 4.0) * 0.28 * k.x + vnoise(y * 26.0 + 9.0) * 0.12 * k.y; }

// magma ramp: t = 0 gap center (hot) -> 1 gap margin (dark)
vec3 magmaRamp(float t, float heat) {
  t = clamp(t + (1.0 - heat) * 0.55, 0.0, 1.0);
  vec3 c = mix(vec3(1.00, 0.93, 0.62), vec3(1.00, 0.68, 0.16), smoothstep(0.00, 0.22, t));
  c = mix(c, vec3(0.95, 0.32, 0.03), smoothstep(0.18, 0.48, t));
  c = mix(c, vec3(0.50, 0.05, 0.01), smoothstep(0.45, 0.78, t));
  c = mix(c, vec3(0.10, 0.01, 0.00), smoothstep(0.72, 1.00, t));
  return c;
}

// ---------- sky: smoke-veiled dusk with a low moon ----------
vec3 moonGlow(vec3 rd) { float m = max(dot(rd, MOONDIR), 0.0); return vec3(1.0, 0.9, 0.74) * (pow(m, 90.0) * 0.22 + pow(m, 12.0) * 0.06); }
vec3 sky(vec3 rd) {
  float h = max(rd.y, 0.0);
  vec3 c = mix(HAZE * 1.08, vec3(0.035, 0.028, 0.03), smoothstep(0.0, 0.42, h));
  vec2 sp = rd.xz / (h + 0.12) * 1.1 + vec2(uTime * 0.012, 0.0);
  float cl = fbm(sp);
  float veil = smoothstep(0.45, 0.8, cl) * smoothstep(0.0, 0.18, h);
  c = mix(c, c * 0.5 + HAZE * 0.06, veil);
  float m = max(dot(rd, MOONDIR), 0.0);
  float disc = smoothstep(0.99984, 0.99991, m);
  c += vec3(1.0, 0.9, 0.74) * disc * 0.85 * (1.0 - 0.6 * veil);
  c += moonGlow(rd) * (1.0 - 0.6 * veil);
  return c;
}
vec3 fogged(vec3 c, float t, vec3 rd) {
  vec3 hz = HAZE * 1.08 + moonGlow(normalize(vec3(rd.x, 0.0, rd.z)));   // = the sky right at the horizon
  return mix(c, hz, 1.0 - exp(-max(t - 8.0, 0.0) * 0.02));
}

// ---------- obsidian bergs: sparse faceted glass shards, at most one per 30-unit cell ----------
const float BC = 30.0;
const vec2 BA[7] = vec2[7](vec2(1.0, 0.0), vec2(0.5254, 0.8508), vec2(-0.1439, 0.9896), vec2(-0.9215, 0.3883),
                           vec2(-0.9398, -0.3418), vec2(-0.1538, -0.9881), vec2(0.5917, -0.8061));
// near-vertical walls (one slightly overhanging): shards, not pyramids
const vec2 BE[7] = vec2[7](vec2(0.9968, 0.0799), vec2(0.9838, 0.1790), vec2(0.9928, 0.1197), vec2(0.9759, 0.2182),
                           vec2(0.9988, -0.0500), vec2(0.9888, 0.1494), vec2(0.9950, 0.0998));
// a berg = a cluster of three leaning glass shards, each with a chisel-cut tip
struct Berg { bool on; vec2 c[3]; vec2 cs[3]; vec2 lean[3]; vec3 cap[3]; float H[3]; float off[21]; };
Berg makeBerg(vec2 cell) {
  Berg b;
  vec2 h = hash2(cell), h2 = hash2(cell + vec2(57.0, 113.0)), h3 = hash2(cell + vec2(211.0, 17.0));
  b.on = h.x < 0.22 && !(abs(cell.x + 0.5) < 1.0 && abs(cell.y + 0.5) < 1.0);
  if (!b.on) return b;   // most cells are empty: skip the shard setup
  vec2 c0 = (cell + 0.5 + (h2 - 0.5) * 0.45) * BC;
  float big = h3.y > 0.86 ? 1.7 : 1.0;     // the odd giant
  for (int i = 0; i < 3; i++) {
    vec2 r = hash2(cell * 4.0 + vec2(float(i), 1.0)), r2 = hash2(cell * 4.0 + vec2(float(i), 2.0));
    float sc = (i == 0 ? 1.0 : 0.45 + 0.4 * r.x) * big;
    float R = (0.9 + 1.1 * h3.x) * sc;
    float yaw = r.y * 6.2832;
    b.cs[i] = vec2(cos(yaw), sin(yaw));
    float a = 6.2832 * r2.x;
    b.c[i] = c0 + (i == 0 ? vec2(0.0) : vec2(cos(a), sin(a)) * (1.4 + 1.2 * r2.y) * min(big, 1.3));
    b.H[i] = (3.5 + 6.0 * h3.x) * sc * (0.7 + 0.6 * r2.y);
    float la = a + 0.6 * (r.x - 0.5);
    b.lean[i] = vec2(cos(la), sin(la)) * min(0.12 + 0.25 * r2.y, 3.5 / b.H[i]);
    float ca = 6.2832 * hash1(cell * 4.0 + vec2(float(i), 3.0)), ct = 0.55 + 0.4 * r.x;
    b.cap[i] = vec3(sin(ct) * cos(ca), cos(ct), sin(ct) * sin(ca));
    for (int k = 0; k < 7; k++) b.off[i * 7 + k] = R * (0.7 + 0.6 * hash1(cell * 32.0 + vec2(float(i * 7 + k), 9.0)));
  }
  return b;
}
vec3 shardQ(Berg b, int i, vec3 p) { vec3 q = vec3(p.x - b.c[i].x, p.y, p.z - b.c[i].y); q.xz -= b.lean[i] * q.y; return q; }
// largest and second-largest plane distance (their difference is small along a facet edge)
vec2 shard2(Berg b, int i, vec3 q) {
  float a = dot(q, b.cap[i]) - b.H[i] * b.cap[i].y, s = -1e9;
  vec2 cs = b.cs[i];
  for (int k = 0; k < 7; k++) {
    vec2 r = vec2(BA[k].x * cs.x - BA[k].y * cs.y, BA[k].x * cs.y + BA[k].y * cs.x);
    float v = dot(q, vec3(BE[k].x * r.x, BE[k].y, BE[k].x * r.y)) - b.off[i * 7 + k];
    if (v > a) { s = a; a = v; } else if (v > s) s = v;
  }
  return vec2(a, s);
}
float bergSD(Berg b, vec3 p) {
  float d = 1e9;
  for (int i = 0; i < 3; i++) d = min(d, shard2(b, i, shardQ(b, i, p)).x);
  return d;
}
float bergEdge(Berg b, vec3 p) {
  vec2 best = vec2(1e9, 0.0);
  for (int i = 0; i < 3; i++) { vec2 e = shard2(b, i, shardQ(b, i, p)); if (e.x < best.x) best = e; }
  return best.x - best.y;
}
// march cell by cell: empty cells are skipped in one step
float traceBergs(vec3 ro, vec3 rd, float tmax, out Berg hb) {
  float t = 0.0;
  vec2 lastCell = vec2(1e9);
  Berg b = makeBerg(floor(ro.xz / BC));
  for (int i = 0; i < 110; i++) {
    if (t > tmax) break;
    vec3 p = ro + rd * t;
    if (p.y > 26.0 && rd.y >= 0.0) break;
    vec2 cell = floor(p.xz / BC);
    if (cell != lastCell) { b = makeBerg(cell); lastCell = cell; }
    vec2 cmin = cell * BC;
    float tx = rd.x > 1e-5 ? (cmin.x + BC - p.x) / rd.x : (rd.x < -1e-5 ? (cmin.x - p.x) / rd.x : 1e9);
    float tz = rd.z > 1e-5 ? (cmin.y + BC - p.z) / rd.z : (rd.z < -1e-5 ? (cmin.y - p.z) / rd.z : 1e9);
    float d = b.on ? bergSD(b, p) : 1e9;
    if (d < 0.0012 * t + 0.002) { hb = b; return t; }
    t += min(d * 0.8, min(tx, tz) + 0.02);
  }
  hb = b;
  return 1e9;
}
vec3 skyClear(vec3 rd) {   // sky without the cloud veil: what polished glass mirrors cleanly
  float h = max(rd.y, 0.0);
  vec3 c = mix(HAZE * 1.08, vec3(0.035, 0.028, 0.03), smoothstep(0.0, 0.42, h));
  float m = max(dot(rd, MOONDIR), 0.0);
  return c + moonGlow(rd) + vec3(1.0, 0.9, 0.74) * smoothstep(0.99984, 0.99991, m) * 0.85;
}
vec3 shadeBerg(Berg b, vec3 p, vec3 rd, vec3 L, float t) {
  float e = 0.002 + 0.0008 * t;
  vec2 k = vec2(e, 0.0);
  vec3 n = normalize(vec3(bergSD(b, p + k.xyy) - bergSD(b, p - k.xyy),
                          bergSD(b, p + k.yxy) - bergSD(b, p - k.yxy),
                          bergSD(b, p + k.yyx) - bergSD(b, p - k.yyx)));
  // faint conchoidal scallops: curved shell bands across each facet
  float fade = 1.0 / (1.0 + t * 0.04);
  float sc = sin(length(p.xz * 1.3 + p.y * 0.6) * 9.0 + vnoise(p.xy * 0.8) * 6.0);
  vec3 g = vec3(vnoise(p.yz * 0.9), vnoise(p.zx * 0.9 + 3.0), vnoise(p.xy * 0.9 + 7.0)) - 0.5;
  float ndv0 = max(dot(n, -rd), 0.0);   // no ripple noise at grazing angles: it only sparkles there
  n = normalize(n + (0.07 * g + 0.025 * sc * normalize(g + 1e-3)) * fade * smoothstep(0.05, 0.35, ndv0));
  float ndv = max(dot(n, -rd), 0.0);
  float fres = 0.07 + 0.93 * pow(1.0 - ndv, 4.0);
  vec3 r = reflect(rd, n);
  vec3 env;
  if (r.y > 0.0) env = skyClear(r) * 1.2;
  else {   // the crust below, with flecks of its glowing seams caught in the glass
    vec2 ur = r.xz / max(-r.y, 0.06) * 0.7 + p.xz * 0.08;
    float fleck = smoothstep(0.8, 0.88, vnoise(ur * vec2(7.0, 1.6))) * smoothstep(-0.7, -0.05, r.y) * fade;
    env = mix(vec3(0.02, 0.01, 0.008), vec3(0.18, 0.05, 0.015), smoothstep(-0.4, 0.0, r.y)) + vec3(0.9, 0.25, 0.04) * fleck * 0.3;
  }
  float rl = max(dot(r, L), 0.0);
  vec3 col = vec3(0.006, 0.0055, 0.007) * (MOON * max(dot(n, L), 0.0) + AMB);
  col += env * fres * 1.25 + MOON * (pow(rl, 500.0) * 5.0 + pow(rl, 30.0) * 0.18);
  col += vec3(0.7, 0.18, 0.03) * exp(-p.y * 1.4) * (0.4 + 0.75 * max(-n.y, 0.0)) * 0.2;   // molten moat below
  // moon-side rim so silhouettes read against the haze
  col += MOON * pow(1.0 - ndv, 3.0) * max(dot(n, L), 0.0) * 0.5;
  // the bow lantern glinting in the black glass
  vec3 lw = uBoatPos + uBoatRot * (LANT_L * BS) - p;
  float ld = length(lw), lr = max(dot(r, lw / ld), 0.0);
  col += LANTC * (pow(lr, 350.0) * 5.0 + pow(lr, 25.0) * 0.12) / (1.0 + ld * ld * 0.04);
  // knife-sharp facet edges catch the light
  float pxw = 0.0012 * t + 0.002;
  float edge = 1.0 - smoothstep(pxw, pxw + 0.05, bergEdge(b, p));
  col += (HAZE * 0.9 + moonGlow(r) * 0.8 + vec3(0.06, 0.05, 0.05)) * edge * mix(0.5, 1.0, fade);
  return col;
}

// ---------- boat: a Victorian paddle-steamer skiff on levitation orbs ----------
float sdEll(vec3 p, vec3 r) { float k0 = length(p / r), k1 = length(p / (r * r)); return k0 * (k0 - 1.0) / k1; }
float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float sdCap(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}
float sdCylY(vec3 p, float r, float y0, float y1) {
  vec2 d = vec2(length(p.xz) - r, abs(p.y - 0.5 * (y0 + y1)) - 0.5 * (y1 - y0));
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
float sdTorusY(vec3 p, float R, float r) { return length(vec2(length(p.xz) - R, p.y)) - r; }
float deckY(float z) { float s = z / 1.45; return 0.10 + 0.11 * s * s + 0.05 * max(s, 0.0); }
const vec3 MAST_TOP = vec3(0.0, 1.3, 0.58);
const vec3 PB = vec3(0.6, 0.09, 0.02);   // paddle box centre (mirrored in x)

// mat: 0 wood, 1 wheelhouse, 2 funnel iron, 3 keel blade, 4 brass, 5 orb, 6 lantern, 7 paddle box, 8 pennant, 9 rigging
float boatSDF(vec3 p, out float mat) {
  float taper = 1.0 - 0.55 * smoothstep(0.0, 1.45, p.z) - 0.22 * smoothstep(0.6, 1.45, -p.z);
  float body = sdEll(vec3(p.x / taper, p.y - 0.12, p.z), vec3(0.52, 0.46, 1.48)) * taper;
  float dy = deckY(p.z);
  float hull = max(body, p.y - dy - 0.065);
  hull = max(hull, -max(body + 0.035, dy - p.y));          // hollow bulwark around the deck
  float d = hull; mat = 0.0;
  float bs = sdCap(p, vec3(0.0, deckY(1.25), 1.25), vec3(0.0, deckY(1.25) + 0.24, 1.9), 0.022);
  if (bs < d) { d = bs; mat = 0.0; }
  // wheelhouse
  vec3 c = p - vec3(0.0, deckY(-0.45) + 0.14, -0.45);
  float cab = sdBox(c, vec3(0.19, 0.13, 0.25)) - 0.025;
  float roof = sdBox(c - vec3(0.0, 0.16, 0.0), vec3(0.23, 0.012, 0.29)) - 0.01;
  if (cab < d) { d = cab; mat = 1.0; }
  if (roof < d) { d = roof; mat = 4.0; }
  // tall raked funnel with a flared brass crown
  vec3 f = p - vec3(0.0, 0.0, 0.1); f.z += 0.12 * (p.y - 0.2);
  float fun = sdCylY(f, 0.075 + 0.04 * smoothstep(0.86, 1.05, f.y), 0.05, 1.05);
  fun = max(fun, -sdCylY(f, 0.07, 0.95, 1.2));               // open mouth
  if (fun < d) { d = fun; mat = (f.y > 0.88 || f.y < 0.2) ? 4.0 : 2.0; }
  // paddle boxes: half-drum housings on each beam
  vec3 pb = vec3(abs(p.x) - PB.x, p.y - PB.y, p.z - PB.z);
  vec2 pd = vec2(length(pb.yz) - 0.27, abs(pb.x) - 0.07);
  float box = min(max(pd.x, pd.y), 0.0) + length(max(pd, 0.0)) - 0.01;
  box = max(box, -pb.y - 0.1);
  if (box < d) { d = box; mat = 7.0; }
  // mast, yard, rigging and a pennant
  float mast = sdCap(p, vec3(0.0, deckY(0.58), 0.58), MAST_TOP, 0.02);
  float yard = sdCap(p, vec3(-0.3, 1.02, 0.58), vec3(0.3, 1.02, 0.58), 0.012);
  if (min(mast, yard) < d) { d = min(mast, yard); mat = 0.0; }
  float rig = min(sdCap(p, MAST_TOP, vec3(0.0, deckY(1.25) + 0.24, 1.9), 0.006),
                  sdCap(p, MAST_TOP - vec3(0.0, 0.05, 0.0), vec3(0.0, deckY(-1.3) + 0.06, -1.3), 0.006));
  rig = min(rig, sdCap(vec3(abs(p.x), p.yz), vec3(0.3, 1.02, 0.58), vec3(0.48, deckY(0.58) + 0.06, 0.58), 0.005));
  if (rig < d) { d = rig; mat = 9.0; }
  vec3 pn = p - MAST_TOP - vec3(0.0, -0.05, 0.0);
  float wav = 0.035 * sin(pn.z * 14.0 - uTime * 7.0) * smoothstep(0.0, 0.35, -pn.z);
  float pen = sdBox(vec3(pn.x - wav, pn.y, pn.z + 0.17), vec3(0.004, 0.035 * (1.0 + pn.z / 0.4), 0.17));
  if (pen < d) { d = pen; mat = 8.0; }
  // bow lantern on its post
  float post = sdCap(p, vec3(0.0, deckY(1.02), 1.02), LANT_L - vec3(0.0, 0.05, 0.0), 0.012);
  if (post < d) { d = post; mat = 4.0; }
  float lan = length(p - LANT_L) - 0.045;
  if (lan < d) { d = lan; mat = 6.0; }
  // four levitation orbs in brass rings
  vec3 o = vec3(abs(p.x) - 0.3, p.y + 0.3, abs(p.z + 0.02) - 0.55);
  float ring = min(sdTorusY(o, 0.115, 0.016), sdCap(o, vec3(0.0), vec3(0.0, 0.22, 0.0), 0.016));
  if (ring < d) { d = ring; mat = 4.0; }
  float orb = length(o) - 0.09;
  if (orb < d) { d = orb; mat = 5.0; }
  // deep swept keel blade at the stern
  float fy = clamp((-p.y - 0.2) / 1.3, 0.0, 1.0);
  float zc = -0.75 - 0.5 * fy;
  float ch = mix(0.3, 0.13, fy);
  vec3 q = vec3(p.x, p.y + 0.85, p.z - zc);
  float th = 0.03 * (0.45 + 0.55 * smoothstep(-ch, ch * 0.6, q.z));
  float fin = sdBox(q, vec3(th, 0.65, ch)) - 0.008;
  if (fin < d) { d = fin; mat = 3.0; }
  return d;
}
float boatMap(vec3 p) { float m; return boatSDF(p, m); }
vec3 boatNormal(vec3 p) {
  const vec2 k = vec2(1.0, -1.0);
  const float e = 0.0015;
  return normalize(k.xyy * boatMap(p + k.xyy * e) + k.yyx * boatMap(p + k.yyx * e)
                 + k.yxy * boatMap(p + k.yxy * e) + k.xxx * boatMap(p + k.xxx * e));
}
// world-space march; the boat model is evaluated in local units scaled by BS
bool traceBoat(vec3 ro, vec3 rd, mat3 invR, out float t, out float tNear, out float dNear) {
  t = 1e9; tNear = 1e9; dNear = 1e9;
  vec3 C = uBoatPos + uBoatRot * (vec3(0.0, -0.2, 0.2) * BS);
  vec3 oc = ro - C;
  float rad = 2.05 * BS;
  float b = dot(oc, rd), c = dot(oc, oc) - rad * rad, h = b * b - c;
  if (h < 0.0) return false;
  h = sqrt(h);
  float tt = max(-b - h, 0.0), tmax = -b + h;
  vec3 lro = invR * (ro - uBoatPos), lrd = invR * rd;
  for (int i = 0; i < 120; i++) {
    float d = boatMap((lro + lrd * tt) / BS) * BS;
    if (d < dNear) { dNear = d; tNear = tt; }
    if (d < 0.0005 * tt) { t = tt; return true; }
    tt += d * 0.75;
    if (tt > tmax) break;
  }
  return false;
}
vec3 shadeBoat(vec3 pw, vec3 rd, mat3 invR, vec3 L) {
  vec3 lp = invR * (pw - uBoatPos) / BS;
  float mat; boatSDF(lp, mat);
  vec3 nl = boatNormal(lp);
  vec3 n = uBoatRot * nl;
  float ndv = max(dot(n, -rd), 0.0);
  float fres = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
  vec3 base = vec3(0.0), em = vec3(0.0);
  float spP = 20.0, spK = 0.12, refl = 0.15;
  bool metal = false;
  float flick = 0.88 + 0.12 * vnoise(vec2(uTime * 7.0, 3.0));
  const vec3 BRASS = vec3(0.66, 0.48, 0.2);
  if (mat < 0.5) {
    float dy = deckY(lp.z);
    if (nl.y > 0.75 && lp.y < dy + 0.015) {                       // deck planks along the length
      float seam = smoothstep(0.0, 0.12, abs(fract(lp.x * 11.0) - 0.5) * 2.0);
      base = vec3(0.26, 0.18, 0.11) * (0.7 + 0.45 * vnoise(vec2(floor(lp.x * 11.0) * 3.7, lp.z * 2.0))) * mix(0.45, 1.0, seam);
    } else if (lp.y > dy + 0.04 && lp.y < dy + 0.1) {             // brass rail cap
      base = BRASS; metal = true;
    } else if (lp.y < -0.1 + 0.05 * lp.z) {                       // copper sheathing with verdigris
      base = mix(vec3(0.34, 0.17, 0.08), vec3(0.12, 0.25, 0.19), smoothstep(0.5, 0.8, vnoise(lp.zy * 7.0)));
      spP = 40.0; spK = 0.35; refl = 0.3;
    } else {                                                       // mahogany hull planks, a cream band at the sheer
      float pl = abs(fract(lp.y * 14.0) - 0.5) * 2.0;
      base = vec3(0.19, 0.08, 0.045) * (0.75 + 0.45 * vnoise(vec2(lp.z * 2.5, floor(lp.y * 14.0) * 3.1)))
           * mix(0.5, 1.0, smoothstep(0.0, 0.15, pl));
      base = mix(base, vec3(0.5, 0.44, 0.33), (1.0 - smoothstep(0.012, 0.02, abs(lp.y - dy + 0.03))) * step(lp.y, dy + 0.04));
      spP = 30.0; spK = 0.18;
    }
  } else if (mat < 1.5) {                                         // green wheelhouse with lit windows
    vec3 c = lp - vec3(0.0, deckY(-0.45) + 0.14, -0.45);
    base = vec3(0.05, 0.1, 0.07); spK = 0.25;
    float wy = 1.0 - smoothstep(0.035, 0.05, abs(c.y - 0.02));
    float side = step(0.6, abs(nl.x)) * (1.0 - step(0.55, abs(fract(c.z * 4.2 + 0.5) - 0.5) * 2.0));
    float ends = step(0.6, abs(nl.z)) * (1.0 - smoothstep(0.09, 0.11, abs(c.x)));
    float win = wy * max(side, ends);
    base = mix(base, vec3(0.05), win);
    em += vec3(1.0, 0.55, 0.18) * 1.3 * win * flick;
  } else if (mat < 2.5) {
    base = vec3(0.035, 0.033, 0.032); spP = 40.0; spK = 0.3; refl = 0.2;
  } else if (mat < 3.5) {
    base = vec3(0.06, 0.06, 0.065); spP = 40.0; spK = 0.3;
    float hot = 1.0 - smoothstep(-0.05, 0.32, pw.y);
    em += magmaRamp(1.0 - 0.9 * hot, 1.0) * hot * 1.1;
  } else if (mat < 4.5) {
    base = BRASS; metal = true;
    // embers glowing inside the funnel mouth
    vec3 f = lp - vec3(0.0, 0.0, 0.1); f.z += 0.12 * (lp.y - 0.2);
    if (f.y > 0.95 && length(f.xz) < 0.072) em += vec3(1.0, 0.4, 0.08) * 1.6 * flick;
  } else if (mat < 5.5) {                                          // levitation orb: swirling aether
    float sw = vnoise(nl.xz * 3.0 + vec2(uTime * 0.7, -uTime * 0.4));
    em += AETHER * (1.2 + 0.9 * sw) * (0.6 + 0.4 * ndv);
  } else if (mat < 6.5) {
    em += LANTC * 3.0 * flick;
  } else if (mat < 7.5) {                                          // paddle box: oxblood with brass rim and fan vents
    vec3 pb = vec3(abs(lp.x) - PB.x, lp.y - PB.y, lp.z - PB.z);
    float rr = length(pb.yz), ang = atan(pb.y, pb.z);
    base = vec3(0.24, 0.05, 0.04); spK = 0.3;
    if (abs(nl.x) > 0.6) {
      float vent = step(0.5, fract(ang * 9.0 / 6.2832)) * step(0.1, rr) * step(rr, 0.21);
      base = mix(base, vec3(0.02, 0.015, 0.012), vent);
      if (rr > 0.235 || rr < 0.07) { base = BRASS; metal = true; }
    } else { base = mix(base, BRASS, 0.0); }
  } else if (mat < 8.5) {
    base = vec3(0.5, 0.06, 0.04); spK = 0.05;
  } else {
    base = vec3(0.09, 0.075, 0.06);
  }
  if (metal) { spP = 60.0; spK = 1.0; refl = 0.75; }
  float diff = max(dot(n, L), 0.0);
  float sp = pow(max(dot(n, normalize(L - rd)), 0.0), spP) * spK;
  float up = max(-n.y, 0.0);
  vec3 lightIn = MOON * diff + AMB * 1.8;
  lightIn += vec3(1.0, 0.36, 0.08) * (0.15 + 1.2 * uUpGlow) * (0.2 + 0.8 * up);       // magma below
  float dF = length(pw.xz - uFinPos);
  lightIn += vec3(1.0, 0.45, 0.1) * exp(-dF * dF / 0.3) * (1.0 - smoothstep(0.0, 0.8, pw.y)) * 1.1;
  lightIn += AETHER * 0.35 * up;                                                       // orbs underneath
  vec3 lw = uBoatPos + uBoatRot * (LANT_L * BS) - pw;
  float ll = length(lw);
  lightIn += LANTC * flick * max(dot(n, lw / ll), 0.0) * 0.6 / (1.0 + ll * ll * 3.0);  // bow lantern
  vec3 tint = metal ? base * 2.0 : vec3(1.0);
  vec3 col = base * lightIn + MOON * sp * tint;
  vec3 rdir = reflect(rd, n);
  vec3 env = rdir.y > 0.0 ? sky(rdir) : vec3(0.26, 0.07, 0.02) * (0.4 + uUpGlow);
  col += env * fres * refl * tint;
  if (metal) col += env * 0.25 * base;   // polished brass always picks up the warm world
  return col + em;
}

// ---------- crust shading ----------
vec3 crustBase(vec3 ro, vec3 rd, float tp, mat3 invR, vec3 L, float pixAng, bool hitB, float tb) {
  vec3 P = ro + rd * tp;
  float sinG = max(-rd.y, 1e-3);
  float pxW = tp * pixAng * 0.5 * (1.0 + 1.0 / max(sinG, 0.1));   // pixel footprint on the crust
  float pxL = pxW * S;
  float lod = smoothstep(0.14, 0.4, pxL);
  vec4 F0 = fieldTex(P.xz);
  float I = F0.r;
  float heat = smoothstep(0.05, 0.7, I);

  // boat light on the crust: long moon shadow, aether pools under the orbs, lantern pool, glow at the fin
  vec3 ps = P + L * ((uBoatPos.y - P.y) / L.y);
  vec3 lbs = invR * (ps - uBoatPos) / BS;
  float vis = 1.0 - 0.7 * exp(-(lbs.x * lbs.x / 0.22 + lbs.z * lbs.z / 1.6));
  vec3 lb = invR * (P - uBoatPos) / BS;
  vec2 oq = vec2(abs(lb.x) - 0.3, abs(lb.z + 0.02) - 0.55);
  float pool = exp(-dot(oq, oq) / 0.12) * 0.55 + exp(-(lb.x * lb.x / 0.5 + lb.z * lb.z / 1.6)) * 0.16;
  vec3 lw = uBoatPos + uBoatRot * (LANT_L * BS) - P;
  float l2 = dot(lw, lw);
  float flick = 0.88 + 0.12 * vnoise(vec2(uTime * 7.0, 3.0));
  vec2 dfp = P.xz - uFinPos;
  vec3 addL = AETHER * pool * (0.9 + 0.1 * sin(uTime * 2.3))
            + LANTC * flick * 0.55 * (lw.y / sqrt(l2)) / (1.0 + l2 * 0.8)
            + vec3(1.0, 0.42, 0.08) * exp(-dot(dfp, dfp) / 0.12) * 0.8;

  vec3 avgRock = vec3(0.075, 0.062, 0.054) * (MOON * 0.14 * vis + AMB + addL)
              + MOON * 0.22 * vis * pow(max(dot(reflect(rd, vec3(0.0, 1.0, 0.0)), L), 0.0), 18.0);   // moon path
  float vS = length(fSplit(F0));
  vec3 avg = mix(avgRock + vec3(0.045, 0.007, 0.0), magmaRamp(0.35, heat), clamp(vS * S * 0.9, 0.0, 0.6));
  if (lod >= 1.0) return avg;

  vec2 u = P.xz * S;
  vec2 Wl = warpLow(u);
  const float e = 0.05;
  vec2 Jx = (warpLow(u + vec2(e, 0.0)) - Wl) / e, Jy = (warpLow(u + vec2(0.0, e)) - Wl) / e;
  mat2 M = S * mat2(vec2(1.0, 0.0) + Jx, vec2(0.0, 1.0) + Jy);   // d(lattice)/d(world)
  vec2 X = u + Wl + warpHi(u);
  bool live = I > 0.004;
  Q q = plates(X, P.xz, M, live);
  float G = max(q.sd1 + max(q.sd2, 0.0), 1e-5);   // local gap width (lattice)
  float cover = 1.0 - smoothstep(-pxL, pxL, q.sd1);

  // --- rock surface, textured in plate-local coordinates so it rides with the plate
  vec2 Yl = X - M * q.o1;
  float tone = hash1(floor(q.s1) + 17.0);
  vec2 kb = vec2(1.0 - smoothstep(0.06, 0.14, pxL), 1.0 - smoothstep(0.02, 0.05, pxL));
  float hb = bumpH(Yl, kb), hx = bumpH(Yl + vec2(0.02, 0.0), kb), hz = bumpH(Yl + vec2(0.0, 0.02), kb);
  vec3 nr = normalize(vec3(-(hx - hb) / 0.02 * 0.16, 1.0, -(hz - hb) / 0.02 * 0.16));
  vec3 alb = vec3(0.08, 0.068, 0.06) * (0.6 + 0.8 * tone) * (0.72 + 0.56 * hb);
  float diff = max(dot(nr, L), 0.0);
  float spec = pow(max(dot(nr, normalize(L - rd)), 0.0), 26.0) * 0.24 * mix(1.0, 0.35, smoothstep(0.04, 0.2, pxL));
  vec3 rock = alb * (MOON * diff * vis + AMB + addL) + MOON * spec * vis;

  // cooled seams between resting plates: thin, dim, some almost dark
  float dV = max(-q.sdV1, 0.0) / S;
  float ws = 0.009 + 0.013 * vnoise(X * 1.7);
  float sb = (0.04 + 0.96 * smoothstep(0.45, 0.85, vnoise(X * 0.8 + 3.0))) * (0.85 + 0.15 * sin(uTime * 1.1 + 6.28 * tone));
  float seam = (1.0 - smoothstep(ws - pxW, ws + pxW, dV)) * min(1.0, ws / pxW);
  rock = mix(rock, vec3(0.36, 0.06, 0.01) * sb * (0.55 + 0.45 * clamp(1.0 - dV / ws, 0.0, 1.0)), seam);
  rock += vec3(0.10, 0.025, 0.004) * sb * exp(-dV / 0.045) * 0.4;

  // heat glow beside real gaps
  float dE = max(-q.sd1, 0.0) / S;
  rock += vec3(0.55, 0.13, 0.02) * exp(-dE / 0.045) * heat * smoothstep(0.0, 0.06 * S, G);

  // hairline fractures in hot plates
  if (I > 0.62) {
    float eh = voronoiEdgeFree(Yl * 2.4 + 3.3) / 2.4;
    float wh = smoothstep(0.62, 1.0, I) * 0.026 * (0.4 + vnoise(Yl * 6.0)) * smoothstep(0.42, 0.62, vnoise(Yl * 3.0 + 1.7));
    float hm = (1.0 - smoothstep(wh - pxL, wh + pxL, eh)) * smoothstep(0.5 * pxL, 1.5 * pxL, wh);
    hm *= smoothstep(0.0, 3.0 * pxL, -q.sd1);
    rock = mix(rock, magmaRamp(0.35 + 0.65 * pow(eh / max(wh, 1e-4), 0.6), heat * 0.8), hm);
  }

  // surface crack band along hot boundaries (incl. the short crack running ahead of the fin)
  float bw = smoothstep(0.45, 0.95, I) * 0.02 * S * (0.6 + 0.8 * vnoise(X * 5.0)) * smoothstep(0.3, 0.6, vnoise(X * 2.0 + 4.1));
  float hw = 0.5 * G + bw;
  float dc = abs(q.sd1 - 0.5 * G);
  float band = (1.0 - smoothstep(hw - pxL, hw + pxL, dc)) * smoothstep(0.5 * pxL, 1.5 * pxL, hw);
  vec3 rockSide = mix(rock, magmaRamp(dc / max(hw, 1e-4), heat), band);
  vec3 det = rockSide;

  // --- inside a gap: look down to the magma level; the far plate's wall shows above it
  if (q.sd1 > -pxL) {
    float tm = tp + HM / sinG;
    vec3 P1 = ro + rd * tm;
    vec2 X1 = X + M * (P1.xz - P.xz);
    Q q1 = plates(X1, P1.xz, M, live);
    vec3 gapC;
    if (q1.sd1 > 0.0) {
      float G1 = max(q1.sd1 + max(q1.sd2, 0.0), 1e-5);
      float t1 = abs(q1.sd1 - 0.5 * G1) / (0.5 * G1);
      float gheat = heat * mix(0.75, 1.0, smoothstep(0.0, 0.1 * S, G1));
      float flow = fbm(P1.xz * 2.5 + vec2(uTime * 0.25, -uTime * 0.4));
      gapC = magmaRamp(t1, gheat) * (0.78 + 0.45 * flow) + addL * 0.12;
    } else {
      float f = clamp(q.sd1 / max(q.sd1 - q1.sd1, 1e-4), 0.0, 1.0);   // how deep the ray meets the wall
      gapC = mix(vec3(0.022, 0.013, 0.01), magmaRamp(0.62, heat) * 0.75, pow(f, 1.6));
    }
    if (hitB && tb < tm) gapC = shadeBoat(ro + rd * tb, rd, invR, L);   // the fin, seen down the slot
    det = mix(gapC, rockSide, cover);
  }
  return mix(det, avg, lod);
}

// a molten moat where the crust meets each berg
vec3 crust(vec3 ro, vec3 rd, float tp, mat3 invR, vec3 L, float pixAng, bool hitB, float tb) {
  vec3 col = crustBase(ro, rd, tp, invR, L, pixAng, hitB, tb);
  vec3 P = ro + rd * tp;
  Berg b = makeBerg(floor(P.xz / BC));
  if (b.on) {
    float pxW = tp * pixAng;
    float dB = bergSD(b, vec3(P.x, 0.03, P.z));
    float w = 0.06 + 0.1 * vnoise(P.xz * 2.3);
    w = max(w, 1.5 * pxW);
    float moat = (1.0 - smoothstep(w, w + 0.05 + pxW, dB)) * step(-0.3, dB);
    col = mix(col, magmaRamp(0.25 + 0.6 * clamp(dB / w, 0.0, 1.0), 0.85), moat);
    col += vec3(0.5, 0.12, 0.02) * exp(-max(dB, 0.0) / 0.5) * 0.25;
  }
  return col;
}
vec3 halo(vec3 col, vec3 ro, vec3 rd, vec3 c, float tD, vec3 hc, float k) {
  vec3 oc = c - ro;
  float tc = dot(oc, rd);
  if (tc < 0.0 || tc > tD + 0.12) return col;
  float d2 = max(dot(oc, oc) - tc * tc, 0.0);
  return col + hc * k * (0.0009 / (d2 + 0.0009) + 0.05 * exp(-d2 / 0.3));
}

void main() {
  vec2 ndc = (gl_FragCoord.xy * 2.0 - uRes) / uRes.y;
  vec3 rd = normalize(uCamR * ndc.x * uTanH + uCamU * ndc.y * uTanH + uCamF);
  vec3 ro = uCamPos;
  vec3 L = MOONDIR;
  float pixAng = 2.0 * uTanH / uRes.y;
  mat3 invR = transpose(uBoatRot);

  float tb, tNear, dNear;
  bool hitB = traceBoat(ro, rd, invR, tb, tNear, dNear);
  float tp = rd.y < -1e-4 ? -ro.y / rd.y : 1e9;
  Berg hb;
  float tg = traceBergs(ro, rd, min(tp, 260.0), hb);
  float tS = min(tp, tg);

  // boat coverage with a one-pixel soft edge
  float aB = 0.0, tB = 1e9;
  if (hitB && tb < tS) { aB = 1.0; tB = tb; }
  else if (!hitB && tNear < tS) { aB = 1.0 - smoothstep(0.0, tNear * pixAng, dNear); tB = tNear; }

  vec3 col = vec3(0.0);
  if (aB < 1.0) {
    if (tg < tp) col = fogged(shadeBerg(hb, ro + rd * tg, rd, L, tg), tg * 0.8, rd);   // glass stays darker in the haze
    else if (tp > 1e8) col = sky(rd);
    else col = fogged(crust(ro, rd, tp, invR, L, pixAng, hitB, tb), tp, rd);
  }
  if (aB > 0.0) col = mix(col, fogged(shadeBoat(ro + rd * tB, rd, invR, L), tB, rd), aB);
  float tD = aB > 0.5 ? tB : tS;

  // glow halos (before the smoke, so smoke veils them): the bow lantern and the four orbs
  float flick = 0.88 + 0.12 * vnoise(vec2(uTime * 7.0, 3.0));
  col = halo(col, ro, rd, uBoatPos + uBoatRot * (LANT_L * BS), tD, LANTC, 0.9 * flick);
  for (int i = 0; i < 4; i++) {
    vec3 ol = vec3(i < 2 ? 0.3 : -0.3, -0.3, (i % 2 == 0) ? 0.53 : -0.57) * BS;
    col = halo(col, ro, rd, uBoatPos + uBoatRot * ol, tD, AETHER, 0.25);
  }
  // funnel smoke: soft analytic puffs (sorted far to near in JS), lit warm from below
  for (int i = 0; i < 14; i++) {
    vec4 sm = uSmoke[i];
    if (sm.w < 0.0) continue;
    float r = 0.14 + 1.1 * sm.w;
    vec3 oc = sm.xyz - ro;
    float tc = dot(oc, rd);
    if (tc < 0.0 || tc - r > tD) continue;
    float d2 = max(dot(oc, oc) - tc * tc, 0.0);
    if (d2 > 4.0 * r * r) continue;
    vec3 x = ro + rd * tc - sm.xyz;
    float wisp = 0.55 + 0.9 * (vnoise(x.xy / r * 1.6 + float(i) * 7.3) - 0.5);
    float dens = exp(-d2 / (r * r)) * pow(1.0 - sm.w, 1.4) * 0.85 * wisp;
    dens *= clamp((tD - (tc - r)) / (2.0 * r), 0.0, 1.0);
    vec3 sc = vec3(0.16, 0.145, 0.135) + vec3(0.25, 0.07, 0.02) * 0.35 * (1.0 - sm.w);
    col = mix(col, fogged(sc, tc, rd), clamp(dens, 0.0, 0.85));
  }

  col += (hash1(gl_FragCoord.xy) - 0.5) / 255.0;   // dither the dark gradients
  fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.92)), 1.0);
}
