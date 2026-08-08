/**
 * VisionFeed — webcam preview with a targeting overlay: bounding boxes for
 * detected objects, face/hand counts, an OCR readout, and a scanning-line
 * effect while the feed is live and being analyzed.
 */
import { useEffect, useRef, useState } from 'react';
import type { ReactElement } from 'react';
import { Frame } from '../hud';
import { WebcamController, type WebcamStatus } from '../lib/webcam';
import { useJarvis } from '../state/useJarvis';
import styles from './VisionFeed.module.css';

const SEND_FPS = 1;

export function VisionFeed(): ReactElement {
  const { vision, toggles, actions } = useJarvis();
  const videoRef = useRef<HTMLVideoElement>(null);
  const controllerRef = useRef<WebcamController | null>(null);
  const [status, setStatus] = useState<WebcamStatus>({ state: 'idle', message: 'OPTICAL FEED STANDBY' });

  if (!controllerRef.current) {
    controllerRef.current = new WebcamController({ captureMaxEdge: 512 });
  }

  useEffect(() => {
    const controller = controllerRef.current;
    if (!controller) return;
    const unsub = controller.onStatus(setStatus);
    controller.attach(videoRef.current);
    return () => {
      unsub();
      controller.dispose();
      controllerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const controller = controllerRef.current;
    if (!controller) return;

    if (!toggles.vision) {
      controller.stop();
      return;
    }

    let cancelled = false;
    void controller.start().then((ok) => {
      if (!cancelled && ok) {
        controller.startSending((jpegB64) => actions.sendFrame(jpegB64), SEND_FPS);
      }
    });

    return () => {
      cancelled = true;
      controller.stopSending();
    };
  }, [toggles.vision, actions]);

  const live = status.state === 'live';
  const width = vision?.width ?? 0;
  const height = vision?.height ?? 0;
  const hasScale = width > 0 && height > 0;

  return (
    <Frame title="VISION FEED" status={live ? 'LIVE' : status.state.toUpperCase()} flush>
      <div className={styles.wrap}>
        <video ref={videoRef} className={styles.video} muted playsInline />

        {!live ? <div className={styles.placeholder}>{status.message}</div> : null}

        {live ? (
          <div className={styles.overlay}>
            {live ? <span className={styles.scan} /> : null}
            {hasScale
              ? vision?.objects.map((obj, i) => {
                  const [x, y, w, h] = obj.box;
                  const left = (x / width) * 100;
                  const top = (y / height) * 100;
                  const bw = (w / width) * 100;
                  const bh = (h / height) * 100;
                  return (
                    <div
                      key={`${obj.label}-${i}`}
                      className={styles.box}
                      style={{ left: `${left}%`, top: `${top}%`, width: `${bw}%`, height: `${bh}%` }}
                    >
                      <span className={styles.boxLabel}>
                        {obj.label} {(obj.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                })
              : null}
          </div>
        ) : null}

        {live ? (
          <div className={styles.readout}>
            <div className={styles.statsRow}>
              <span>FACES {vision?.faces ?? 0}</span>
              <span>HANDS {vision?.hands ?? 0}</span>
              <span>OBJECTS {vision?.objects.length ?? 0}</span>
            </div>
            {vision?.text ? <div className={styles.ocr}>{vision.text}</div> : null}
          </div>
        ) : null}
      </div>
    </Frame>
  );
}

export default VisionFeed;
