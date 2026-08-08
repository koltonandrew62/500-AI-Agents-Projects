/**
 * CoreSphere — the centerpiece particle core.
 *
 * ~4000 points distributed on a Fibonacci sphere, displaced per-particle by
 * two octaves of simplex noise in the vertex shader. `intensity` (0–1) drives
 * both displacement amplitude and rotation speed, so voice amplitude or a
 * "thinking" state visibly agitates the core.
 *
 * Lifecycle contract: the renderer, geometry and material are created once per
 * `particleCount` and fully disposed on unmount (including a forced context
 * loss) — mounting/unmounting this component repeatedly must not leak GL
 * contexts. Resize is handled by ResizeObserver, not window listeners.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactElement } from 'react';
import * as THREE from 'three';
import { CORE_FRAGMENT_SHADER, CORE_VERTEX_SHADER } from './coreShaders';
import { approach, clamp, usePrefersReducedMotion } from './hooks';
import styles from './CoreSphere.module.css';

export type CoreState = 'idle' | 'listening' | 'thinking' | 'speaking' | 'error';

export interface CoreSphereProps {
  /** 0–1 agitation level: voice amplitude, or a synthetic pulse while thinking. */
  intensity?: number;
  /** Assistant state — tints the core and biases its base rotation speed. */
  state?: CoreState;
  /** Number of points on the sphere. Default 4000. */
  particleCount?: number;
  /** Base point size in device-independent units. Default 2.6. */
  pointSize?: number;
  className?: string;
}

/** Token-derived palette. Only colors from CONTRACTS.md §6 appear here. */
const STATE_COLORS: Record<CoreState, { edge: string; core: string }> = {
  idle: { edge: '#1c6f92', core: '#7ee8ff' },
  listening: { edge: '#4fd8ff', core: '#7ee8ff' },
  thinking: { edge: '#1b8fe0', core: '#7ee8ff' },
  speaking: { edge: '#4fd8ff', core: '#ffffff' },
  error: { edge: '#ff4d5e', core: '#ffb545' },
};

const STATE_SPIN: Record<CoreState, number> = {
  idle: 0.10,
  listening: 0.22,
  thinking: 0.46,
  speaking: 0.30,
  error: 0.16,
};

/** Evenly distributes `count` points on a unit sphere (golden-angle spiral). */
function buildFibonacciSphere(count: number): { positions: Float32Array; seeds: Float32Array } {
  const positions = new Float32Array(count * 3);
  const seeds = new Float32Array(count);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  for (let i = 0; i < count; i += 1) {
    const y = 1 - (i / Math.max(count - 1, 1)) * 2;
    const radius = Math.sqrt(Math.max(1 - y * y, 0));
    const theta = goldenAngle * i;
    positions[i * 3] = Math.cos(theta) * radius;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = Math.sin(theta) * radius;
    // Deterministic pseudo-random seed — stable across remounts.
    seeds[i] = (Math.sin(i * 127.1) * 43758.5453) % 1;
    if (seeds[i] < 0) seeds[i] += 1;
  }
  return { positions, seeds };
}

interface CoreRuntime {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  group: THREE.Group;
  geometry: THREE.BufferGeometry;
  material: THREE.ShaderMaterial;
  render: () => void;
}

export function CoreSphere({
  intensity = 0,
  state = 'idle',
  particleCount = 4000,
  pointSize = 2.6,
  className,
}: CoreSphereProps): ReactElement {
  const hostRef = useRef<HTMLDivElement>(null);
  const runtimeRef = useRef<CoreRuntime | null>(null);
  const [failed, setFailed] = useState(false);
  const reducedMotion = usePrefersReducedMotion();

  // Latest props live in refs so the animation loop never restarts on prop churn.
  const targetIntensity = clamp(intensity);
  const intensityRef = useRef(targetIntensity);
  intensityRef.current = targetIntensity;
  const spinRef = useRef(STATE_SPIN[state]);
  spinRef.current = STATE_SPIN[state];

  const colors = useMemo(() => STATE_COLORS[state], [state]);
  const count = Math.max(256, Math.floor(particleCount));

  // ---- Scene construction (re-runs only if the particle budget changes) ----
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let runtime: CoreRuntime;
    try {
      const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: false,
        powerPreference: 'high-performance',
      });
      renderer.setClearColor(0x000000, 0);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
      camera.position.set(0, 0, 3.35);

      const { positions, seeds } = buildFibonacciSphere(count);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));

      const material = new THREE.ShaderMaterial({
        vertexShader: CORE_VERTEX_SHADER,
        fragmentShader: CORE_FRAGMENT_SHADER,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uTime: { value: 0 },
          uIntensity: { value: intensityRef.current },
          uSize: { value: pointSize },
          uPixelRatio: { value: renderer.getPixelRatio() },
          uShellBias: { value: 0.07 },
          uOpacity: { value: 0.92 },
          uColorEdge: { value: new THREE.Color(STATE_COLORS.idle.edge) },
          uColorCore: { value: new THREE.Color(STATE_COLORS.idle.core) },
        },
      });

      const points = new THREE.Points(geometry, material);
      const group = new THREE.Group();
      group.add(points);
      scene.add(group);

      host.appendChild(renderer.domElement);
      runtime = {
        renderer,
        scene,
        camera,
        group,
        geometry,
        material,
        render: () => renderer.render(scene, camera),
      };
      runtimeRef.current = runtime;
      setFailed(false);
    } catch {
      setFailed(true);
      return;
    }

    // ---- Sizing ----
    const resize = (): void => {
      const width = Math.max(host.clientWidth, 1);
      const height = Math.max(host.clientHeight, 1);
      runtime.renderer.setSize(width, height, false);
      runtime.camera.aspect = width / height;
      runtime.camera.updateProjectionMatrix();
      runtime.material.uniforms.uPixelRatio.value = runtime.renderer.getPixelRatio();
      runtime.render();
    };
    resize();

    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(resize);
      observer.observe(host);
    } else {
      window.addEventListener('resize', resize);
    }

    // ---- Context-loss resilience ----
    const canvas = runtime.renderer.domElement;
    const onContextLost = (event: Event): void => {
      event.preventDefault();
      setFailed(true);
    };
    canvas.addEventListener('webglcontextlost', onContextLost);

    // ---- Animation ----
    let frame = 0;
    let last = performance.now();
    let elapsed = 0;
    let smoothed = intensityRef.current;

    const tick = (now: number): void => {
      const delta = Math.min((now - last) / 1000, 0.1);
      last = now;

      // Rise fast, fall slow — mirrors how a VU meter feels.
      const target = intensityRef.current;
      const rate = target > smoothed ? 14 : 4.5;
      smoothed = approach(smoothed, target, rate, delta);

      elapsed += delta * (0.6 + smoothed * 1.1);
      runtime.material.uniforms.uTime.value = elapsed;
      runtime.material.uniforms.uIntensity.value = smoothed;

      const spin = spinRef.current + smoothed * 0.45;
      runtime.group.rotation.y += delta * spin;
      runtime.group.rotation.x = Math.sin(elapsed * 0.18) * 0.16;
      runtime.group.rotation.z = Math.cos(elapsed * 0.11) * 0.07;

      runtime.render();
      frame = requestAnimationFrame(tick);
    };

    if (!reducedMotion) {
      frame = requestAnimationFrame(tick);
    } else {
      // Static pose: one representative frame, no continuous motion.
      runtime.material.uniforms.uTime.value = 1.7;
      runtime.material.uniforms.uIntensity.value = intensityRef.current;
      runtime.group.rotation.set(0.16, 0.5, 0);
      runtime.render();
    }

    return () => {
      cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener('resize', resize);
      canvas.removeEventListener('webglcontextlost', onContextLost);

      runtime.geometry.dispose();
      runtime.material.dispose();
      runtime.scene.clear();
      runtime.renderer.dispose();
      runtime.renderer.forceContextLoss();
      if (canvas.parentNode === host) host.removeChild(canvas);
      runtimeRef.current = null;
    };
  }, [count, pointSize, reducedMotion]);

  // ---- Color follows state without rebuilding the scene ----
  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    (runtime.material.uniforms.uColorEdge.value as THREE.Color).set(colors.edge);
    (runtime.material.uniforms.uColorCore.value as THREE.Color).set(colors.core);
    if (reducedMotion) runtime.render();
  }, [colors, reducedMotion]);

  // ---- Reduced motion still reflects intensity, just without a loop ----
  useEffect(() => {
    if (!reducedMotion) return;
    const runtime = runtimeRef.current;
    if (!runtime) return;
    runtime.material.uniforms.uIntensity.value = targetIntensity;
    runtime.render();
  }, [reducedMotion, targetIntensity]);

  const haloOpacity = 0.45 + targetIntensity * 0.55;

  return (
    <div className={`${styles.wrap} ${className ?? ''}`.trim()} aria-hidden="true">
      <div className={styles.halo} style={{ opacity: haloOpacity }} />
      {failed ? (
        <div className={styles.fallback}>core render unavailable</div>
      ) : (
        <div className={styles.canvasHost} ref={hostRef} />
      )}
      <div className={styles.fringe} />
    </div>
  );
}

export default CoreSphere;
