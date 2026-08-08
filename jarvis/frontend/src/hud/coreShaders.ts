/**
 * GLSL for the particle core. Kept out of CoreSphere.tsx so that file stays
 * focused on lifecycle + resource disposal.
 *
 * The noise function is Ashima Arts' 3D simplex noise (MIT), the standard
 * implementation used across the WebGL ecosystem.
 */

const SIMPLEX_3D = /* glsl */ `
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289(i);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}
`;

export const CORE_VERTEX_SHADER = /* glsl */ `
uniform float uTime;
uniform float uIntensity;
uniform float uSize;
uniform float uPixelRatio;
uniform float uShellBias;

attribute float aSeed;

varying float vGlow;
varying float vSeed;

${SIMPLEX_3D}

void main() {
  vSeed = aSeed;
  vec3 dir = normalize(position);

  // Two octaves: a slow rolling swell plus a finer shimmer that speeds up
  // with intensity, so a loud voice visibly agitates the surface.
  float slow = snoise(dir * 2.1 + vec3(0.0, 0.0, uTime * 0.22));
  float fine = snoise(dir * 5.4 - vec3(uTime * (0.30 + uIntensity * 0.85)));

  float breath = 0.5 + 0.5 * sin(uTime * 1.35 + aSeed * 6.2831853);
  float amp = 0.030 + uIntensity * 0.290;

  float disp = (slow * 0.68 + fine * 0.32) * amp
             + uIntensity * 0.045 * breath;

  // Particles sit on a thin shell; aSeed jitters the radius so the surface
  // has depth rather than reading as a hollow cutout.
  float shell = 1.0 + uShellBias * (aSeed - 0.5);
  vec3 displaced = dir * (shell + disp);

  vGlow = clamp(disp / max(amp, 0.0001) * 0.5 + 0.5, 0.0, 1.0);

  vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
  gl_PointSize = uSize * uPixelRatio
               * (0.75 + 0.55 * aSeed)
               * (1.0 + uIntensity * 0.55)
               * (300.0 / max(-mvPosition.z, 0.001));
  gl_Position = projectionMatrix * mvPosition;
}
`;

export const CORE_FRAGMENT_SHADER = /* glsl */ `
precision mediump float;

uniform vec3 uColorEdge;
uniform vec3 uColorCore;
uniform float uOpacity;

varying float vGlow;
varying float vSeed;

void main() {
  vec2 uv = gl_PointCoord - vec2(0.5);
  float dist = length(uv);
  if (dist > 0.5) discard;

  // Soft radial falloff — a hard disc looks like confetti, not plasma.
  float alpha = pow(smoothstep(0.5, 0.0, dist), 2.4);

  // Cyan at the resting radius, white-hot where displacement pushes outward.
  vec3 color = mix(uColorEdge, uColorCore, pow(vGlow, 1.6));

  float twinkle = 0.72 + 0.28 * vSeed;
  gl_FragColor = vec4(color, alpha * uOpacity * twinkle * (0.42 + 0.58 * vGlow));
}
`;
