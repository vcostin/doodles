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

#define MAX_PTS 96
uniform vec2  uOrg;      // world xz of the window's corner
uniform float uSize;     // window size (world units)
uniform float uRes;      // texels across
uniform float uTime;
uniform vec4  uPts[MAX_PTS];  // xy: world xz, z: life (heat), w: split amount (world)
uniform vec4  uAux[MAX_PTS];  // x: reach (world), yz: heading
uniform int   uCount;         // chronological; the last point is the live fin position
out vec4 fragColor;

const float SR = 0.4, DR = 2.5;   // encoding ranges: split vector, signed distance

void main() {
  vec2 p = uOrg + gl_FragCoord.xy / uRes * uSize;
  float wob = vnoise(p * 1.7 + vec2(uTime * 0.15, 0.0)) - 0.5;
  vec4 head = uPts[max(uCount - 1, 0)];
  vec2 headDir = uAux[max(uCount - 1, 0)].yz;
  float bestScore = -1.0, split = 0.0, sdist = DR, I = 0.0;
  vec2 nrm = vec2(0.0);
  for (int i = 0; i < MAX_PTS; i++) {
    if (i >= uCount) break;
    bool last = i + 1 >= uCount;
    vec4 A = uPts[i], Aa = uAux[i];
    vec4 B = last ? A : uPts[min(i + 1, MAX_PTS - 1)];
    vec4 Ba = last ? Aa : uAux[min(i + 1, MAX_PTS - 1)];
    vec2 ab = B.xy - A.xy;
    float L2 = dot(ab, ab);
    if (L2 > 1.0) { B = A; Ba = Aa; ab = vec2(0.0); L2 = 0.0; }   // broken trail: never bridge a jump
    float h = L2 > 1e-8 ? clamp(dot(p - A.xy, ab) / L2, 0.0, 1.0) : 0.0;
    vec2 dir = L2 > 1e-8 ? ab * inversesqrt(L2) : Aa.yz;
    vec2 d = p - (A.xy + ab * h);
    float dist = length(d);
    float lf = mix(A.z, B.z, h), amt = mix(A.w, B.w, h), R = mix(Aa.x, Ba.x, h);
    // crust in front of the fin stays (almost) whole: only a short crack runs ahead
    float aheadF = 1.0;
    if (i >= uCount - 2) { float ah = max(dot(p - head.xy, headDir), 0.0); aheadF = exp(-ah * ah / 0.2); }
    float dd = dist * (1.0 + 0.5 * wob);
    float fall = 1.0 - smoothstep(0.15 * R, R, dd);
    float s = amt * fall * aheadF;
    // strongest segment wins; nearest breaks ties where nothing is moving
    float score = s + 1e-3 * exp(-dist);
    if (score > bestScore) {
      bestScore = score; split = s;
      nrm = vec2(-dir.y, dir.x);
      sdist = (dot(d, nrm) >= 0.0 ? 1.0 : -1.0) * dist;
    }
    I = max(I, lf * (1.0 - smoothstep(0.0, R + 1.2, dd)) * aheadF);
  }
  // fade out at the window border so clamped lookups never smear the wake
  vec2 e2 = min(p - uOrg, uOrg + uSize - p);
  float bf = smoothstep(0.3, 2.0, min(e2.x, e2.y));
  I *= bf; split *= bf;
  fragColor = vec4(I, nrm * split / (2.0 * SR) + 0.5, clamp(sdist, -DR, DR) / (2.0 * DR) + 0.5);
}
