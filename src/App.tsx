/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play, Square, RotateCcw, Download, ZoomIn, ZoomOut, Maximize2,
  Code, Sparkles, Layers, Settings, HelpCircle, CheckCircle2,
  AlertTriangle, Compass, RefreshCw, Copy, Check, Info, FileCode
} from 'lucide-react';

// --- CONSTANTS & CONFIG ---
const CFG = {
  O: [0, 0],
  A: [1, -1],
  B: [0, -1],
  q: (1 + Math.sqrt(5)) / 4,
  colors: { 1: '#ff6b6b', 2: '#4dd0e1', 3: '#ffd166', 4: '#a78bfa' },
  mono: '#c9d4e6'
};

const MAPS = {
  1: (x: number, y: number, a: number, b: number): [number, number] => [-(x + y) / 2 + a, y / 2],
  2: (x: number, y: number, a: number, b: number): [number, number] => [-(x + y) / 2, y / 2 - 1 / 2],
  3: (x: number, y: number, a: number, b: number): [number, number] => [-(x + y) / 2 + 1 / 2, y / 2 - 1 / 2],
  4: (x: number, y: number, a: number, b: number): [number, number] => [(x + y) / 2 + 1 / 2 + b, -y / 2 - 1]
};

function applyMap(i: number, v: [number, number], a: number, b: number): [number, number] {
  return MAPS[i as keyof typeof MAPS](v[0], v[1], a, b);
}

function applyMapToTri(i: number, tri: { v: [number, number][]; a: string; outer: number }, a: number, b: number) {
  return {
    v: [applyMap(i, tri.v[0], a, b), applyMap(i, tri.v[1], a, b), applyMap(i, tri.v[2], a, b)] as [number, number][],
    a: String(i) + tri.a,
    outer: i
  };
}

function seedFromString(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function safeRadius(a: number, b: number): number {
  const q = CFG.q;
  const ts = [[a, 0], [0, -0.5], [0.5, -0.5], [0.5 + b, -1]];
  let B = 0;
  for (const t of ts) {
    B = Math.max(B, Math.hypot(t[0], t[1]));
  }
  return Math.max(Math.SQRT2, B / (1 - q));
}

// --- PYTHON NUMBA CODE TEMPLATE ---
const NUMBA_PYTHON_CODE = `import numpy as np
from numba import njit, prange
import time

# 1. Định nghĩa hàm ánh xạ affine (được tối ưu hóa với Numba @njit)
@njit(fastmath=True)
def affine_map(x, y, i, a, b):
    if i == 0:
        return -(x + y) / 2.0 + a, y / 2.0
    elif i == 1:
        return -(x + y) / 2.0, y / 2.0 - 0.5
    elif i == 2:
        return -(x + y) / 2.0 + 0.5, y / 2.0 - 0.5
    else:
        return (x + y) / 2.0 + 0.5 + b, -y / 2.0 - 1.0

# 2. Thuật toán Chaos Game tăng tốc song song bằng Numba
@njit(fastmath=True)
def generate_chaos_points(n_points, skip, alpha, beta, seed):
    # Khởi tạo mảng lưu tọa độ và chỉ số ánh xạ (x, y, map_id)
    points = np.empty((n_points, 3), dtype=np.float32)
    
    # Khởi tạo bộ sinh số ngẫu nhiên PCG64 / xorshift đơn giản trong numba
    state = seed if seed != 0 else 20261115
    
    def next_rand():
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= (state >> 17)
        state ^= (state << 5) & 0xFFFFFFFF
        return (state & 0xFFFFFFFF) / 4294967296.0

    x, y = 0.0, 0.0
    
    # Bỏ qua các điểm quá độ (transient steps)
    for _ in range(skip):
        r = next_rand()
        i = int(r * 4.0)
        if i > 3: i = 3
        x, y = affine_map(x, y, i, alpha, beta)
        
    # Vòng lặp chính tính attractor
    for k in range(n_points):
        r = next_rand()
        i = int(r * 4.0)
        if i > 3: i = 3
        x, y = affine_map(x, y, i, alpha, beta)
        points[k, 0] = x
        points[k, 1] = y
        points[k, 2] = float(i + 1)
        
    return points

# 3. Hàm kiểm thử tốc độ và xuất ảnh trực tiếp
def benchmark_and_render(n_points=1_000_000, alpha=0.0, beta=0.0):
    print(f"Đang tính {n_points:,} điểm với Numba JIT (alpha={alpha}, beta={beta})...")
    
    # Warm-up Numba compilation
    _ = generate_chaos_points(1000, 100, alpha, beta, 42)
    
    t0 = time.time()
    pts = generate_chaos_points(n_points, 200, alpha, beta, 20261115)
    t1 = time.time()
    
    print(f"Hoàn thành trong {t1 - t0:.4f} giây!")
    return pts

if __name__ == '__main__':
    pts = benchmark_and_render(1_000_000, alpha=0.0, beta=0.0)
`;

export default function App() {
  // --- STATE ---
  const [mode, setMode] = useState<'chaos' | 'tri' | 'numba'>('chaos');
  const [alpha, setAlpha] = useState<number>(0);
  const [beta, setBeta] = useState<number>(0);
  const [linkedMode, setLinkedMode] = useState<boolean>(true);
  const [showContact, setShowContact] = useState<boolean>(false);
  
  // Chaos params
  const [numPts, setNumPts] = useState<number>(300000);
  const [seedStr, setSeedStr] = useState<string>('20261115');
  const [colorMode, setColorMode] = useState<'mono' | 'map'>('mono');
  const [isCalculating, setIsCalculating] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [progressLabel, setProgressLabel] = useState<string>('');

  // Triangle params
  const [level, setLevel] = useState<number>(0);
  const [showStroke, setShowStroke] = useState<boolean>(false);
  const [triData, setTriData] = useState<{ level: number; triangles: any[]; alpha: number; beta: number } | null>(null);
  const [autoRunning, setAutoRunning] = useState<boolean>(false);

  // View state
  const [view, setView] = useState<{ cx: number; cy: number; scale: number }>({ cx: 0.25, cy: -0.5, scale: 380 });
  const [showAxes, setShowAxes] = useState<boolean>(true);
  const [showGrid, setShowGrid] = useState<boolean>(true);
  const [showOrig, setShowOrig] = useState<boolean>(true);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [autoHideLabels, setAutoHideLabels] = useState<boolean>(true);
  const [labelSize, setLabelSize] = useState<number>(12);
  const [selectedPiece, setSelectedPiece] = useState<string | null>(null);
  const [pieceInfo, setPieceInfo] = useState<any[] | null>(null);
  const [autoUpdate, setAutoUpdate] = useState<boolean>(false);
  const [isDirty, setIsDirty] = useState<boolean>(false);

  // Canvas & Worker refs
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chaosPointsRef = useRef<Float32Array | null>(null);
  const autoRunTimerRef = useRef<any>(null);
  const taskIdRef = useRef<number>(0);
  const workerRef = useRef<Worker | null>(null);

  // Copy state for Numba code
  const [copied, setCopied] = useState<boolean>(false);

  // --- HANDLE ALPHA / BETA CHANGES ---
  const handleAlphaChange = (val: number) => {
    setAlpha(val);
    let newBeta = beta;
    if (linkedMode) {
      newBeta = -val / 3;
      setBeta(newBeta);
    }
    markDirty(val, newBeta);
  };

  const handleBetaChange = (val: number) => {
    if (linkedMode) return;
    setBeta(val);
    markDirty(alpha, val);
  };

  const handleLinkToggle = (checked: boolean) => {
    setLinkedMode(checked);
    let newBeta = beta;
    if (checked) {
      newBeta = -alpha / 3;
      setBeta(newBeta);
    }
    markDirty(alpha, newBeta);
  };

  const markDirty = (a: number, b: number) => {
    setIsDirty(true);
    chaosPointsRef.current = null;
    setTriData(null);
    setSelectedPiece(null);
    setPieceInfo(null);
    if (autoUpdate) {
      setTimeout(() => {
        if (mode === 'chaos') startChaos(a, b);
        else if (mode === 'tri') buildTriLevel(level, a, b);
      }, 50);
    }
  };

  // --- WEB WORKER FOR CHAOS GAME ---
  useEffect(() => {
    const workerCode = `
      self.onmessage = function(ev){
        const msg = ev.data;
        if(msg.type === 'chaos'){
          const {taskId, alpha, beta, seed, N, skip} = msg;
          let t = seed >>> 0;
          function rnd(){
            t = (t + 0x6D2B79F5) >>> 0;
            let r = t;
            r = Math.imul(r ^ (r >>> 15), 1 | t);
            r = r ^ (r + Math.imul(r ^ (r >>> 7), 61 | r));
            return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
          }
          let x=0, y=0;
          const CHUNK = 30000;
          let produced = 0;
          for(let k=0; k<skip; k++){
            const i = (rnd()*4)|0;
            let nx, ny;
            if(i===0){ nx = -(x+y)/2 + alpha; ny = y/2; }
            else if(i===1){ nx = -(x+y)/2; ny = y/2 - 0.5; }
            else if(i===2){ nx = -(x+y)/2 + 0.5; ny = y/2 - 0.5; }
            else { nx = (x+y)/2 + 0.5 + beta; ny = -y/2 - 1; }
            x = nx; y = ny;
          }
          function step(){
            const end = Math.min(produced + CHUNK, N);
            const buf = new Float32Array((end-produced)*3);
            let k = 0;
            while(produced < end){
              const i = (rnd()*4)|0;
              let nx, ny;
              if(i===0){ nx = -(x+y)/2 + alpha; ny = y/2; }
              else if(i===1){ nx = -(x+y)/2; ny = y/2 - 0.5; }
              else if(i===2){ nx = -(x+y)/2 + 0.5; ny = y/2 - 0.5; }
              else { nx = (x+y)/2 + 0.5 + beta; ny = -y/2 - 1; }
              x = nx; y = ny;
              buf[k++] = x; buf[k++] = y; buf[k++] = i+1;
              produced++;
            }
            self.postMessage({type:'chunk', taskId, buf: buf.buffer}, [buf.buffer]);
            if(produced < N){
              setTimeout(step, 0);
            } else {
              self.postMessage({type:'done', taskId});
            }
          }
          step();
        }
      };
    `;
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const w = new Worker(URL.createObjectURL(blob));
    workerRef.current = w;

    return () => {
      w.terminate();
    };
  }, []);

  // --- START CHAOS GAME ---
  const startChaos = (currAlpha = alpha, currBeta = beta) => {
    if (!workerRef.current) return;
    taskIdRef.current++;
    const currentTaskId = taskIdRef.current;
    setIsCalculating(true);
    setProgress(0);
    setProgressLabel(`Đang khởi tạo ${numPts.toLocaleString('vi-VN')} điểm...`);

    const seed = seedFromString(seedStr.trim() || '20261115');
    const skip = 250;
    const bufAll = new Float32Array(numPts * 3);
    let produced = 0;

    workerRef.current.onmessage = (ev) => {
      const m = ev.data;
      if (m.taskId !== currentTaskId) return;
      if (m.type === 'chunk') {
        const arr = new Float32Array(m.buf);
        bufAll.set(arr, produced * 3);
        produced += arr.length / 3;
        const frac = produced / numPts;
        setProgress(frac);
        setProgressLabel(`${produced.toLocaleString('vi-VN')} / ${numPts.toLocaleString('vi-VN')} điểm`);
        
        if (produced % 90000 < arr.length / 3 || produced === numPts) {
          chaosPointsRef.current = bufAll.slice(0, produced * 3);
          setIsDirty(false);
          redrawCanvas();
        }
      } else if (m.type === 'done') {
        chaosPointsRef.current = bufAll;
        setIsCalculating(false);
        setIsDirty(false);
        redrawCanvas();
      }
    };

    workerRef.current.postMessage({
      type: 'chaos',
      taskId: currentTaskId,
      alpha: currAlpha,
      beta: currBeta,
      seed,
      N: numPts,
      skip
    });
  };

  const cancelChaos = () => {
    taskIdRef.current++;
    setIsCalculating(false);
    if (autoRunTimerRef.current) clearTimeout(autoRunTimerRef.current);
    setAutoRunning(false);
  };

  // --- TRIANGLE APPROXIMATION ---
  const buildTriLevel = (lvl: number, currAlpha = alpha, currBeta = beta) => {
    let cur = [{ v: [[0, 0], [1, -1], [0, -1]] as [number, number][], a: '', outer: 0 }];
    for (let lv = 1; lv <= lvl; lv++) {
      const next = [];
      for (const t of cur) {
        for (let i = 1; i <= 4; i++) {
          next.push(applyMapToTri(i, t, currAlpha, currBeta));
        }
      }
      cur = next;
    }
    const data = { level: lvl, triangles: cur, alpha: currAlpha, beta: currBeta };
    setTriData(data);
    setIsDirty(false);
    return data;
  };

  const handleSetLevel = (lvl: number) => {
    setLevel(lvl);
    buildTriLevel(lvl);
  };

  // --- RENDERING CANVAS ---
  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0a0c10';
    ctx.fillRect(0, 0, width, height);

    const W = width / (window.devicePixelRatio || 1);
    const H = height / (window.devicePixelRatio || 1);

    const worldToScreen = (x: number, y: number) => [
      (x - view.cx) * view.scale + W / 2,
      H / 2 - (y - view.cy) * view.scale
    ];

    // 1. Grid & Axes
    if (showGrid || showAxes) {
      const [xmin, ymax] = [ (0 - W / 2) / view.scale + view.cx, view.cy - (0 - H / 2) / view.scale ];
      const [xmax, ymin] = [ (W - W / 2) / view.scale + view.cx, view.cy - (H - H / 2) / view.scale ];
      const range = Math.max(xmax - xmin, ymax - ymin);
      const target = 80 / view.scale;
      const pow = Math.pow(10, Math.floor(Math.log10(target)));
      const n = target / pow;
      let step = pow;
      if (n >= 1.5 && n < 3.5) step = 2 * pow;
      else if (n >= 3.5 && n < 7.5) step = 5 * pow;
      else if (n >= 7.5) step = 10 * pow;

      if (showGrid) {
        ctx.strokeStyle = '#243047';
        ctx.lineWidth = 1;
        ctx.beginPath();
        let x0 = Math.floor(xmin / step) * step;
        for (let x = x0; x <= xmax; x += step) {
          const [sx] = worldToScreen(x, 0);
          ctx.moveTo(sx, 0); ctx.lineTo(sx, H);
        }
        let y0 = Math.floor(ymin / step) * step;
        for (let y = y0; y <= ymax; y += step) {
          const [, sy] = worldToScreen(0, y);
          ctx.moveTo(0, sy); ctx.lineTo(W, sy);
        }
        ctx.stroke();
      }

      if (showAxes) {
        ctx.strokeStyle = '#5a6b8c';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        const [, sy0] = worldToScreen(0, 0);
        const [sx0] = worldToScreen(0, 0);
        ctx.moveTo(0, sy0); ctx.lineTo(W, sy0);
        ctx.moveTo(sx0, 0); ctx.lineTo(sx0, H);
        ctx.stroke();

        // Draw coordinate numbers along axes for clear reference
        ctx.fillStyle = '#a0aec0';
        ctx.font = '11px ui-monospace, Menlo, Consolas, monospace';

        // X axis labels
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        let x0 = Math.floor(xmin / step) * step;
        for (let x = x0; x <= xmax; x += step) {
          if (Math.abs(x) < 1e-6) continue;
          const [sx] = worldToScreen(x, 0);
          const [, sy] = worldToScreen(0, 0);
          const labelY = Math.min(H - 18, Math.max(4, sy + 6));
          ctx.fillText(x.toFixed(2).replace(/\.?0+$/, ''), sx, labelY);
        }

        // Y axis labels
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        let y0 = Math.floor(ymin / step) * step;
        for (let y = y0; y <= ymax; y += step) {
          if (Math.abs(y) < 1e-6) continue;
          const [sx] = worldToScreen(0, 0);
          const [, sy] = worldToScreen(0, y);
          const labelX = Math.max(34, Math.min(W - 10, sx - 8));
          ctx.fillText(y.toFixed(2).replace(/\.?0+$/, ''), labelX, sy);
        }

        // Origin (0,0) label
        const [originSx, originSy] = worldToScreen(0, 0);
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText('0', originSx - 5, originSy + 5);
      }
    }

    // 2. Original Triangle OAB
    if (showOrig) {
      const O = worldToScreen(CFG.O[0], CFG.O[1]);
      const A = worldToScreen(CFG.A[0], CFG.A[1]);
      const B = worldToScreen(CFG.B[0], CFG.B[1]);
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = '#8892a6';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(O[0], O[1]); ctx.lineTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#c9d4e6';
      ctx.font = '12px system-ui';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('O', O[0] - 12, O[1] - 12);
      ctx.fillText('A', A[0] + 12, A[1] + 12);
      ctx.fillText('B', B[0] - 12, B[1] - 12);
      ctx.restore();
    }

    // 3. Chaos Game Points
    if (mode === 'chaos' && chaosPointsRef.current) {
      const pts = chaosPointsRef.current;
      const n = pts.length / 3;
      const imgData = ctx.createImageData(width, height);
      const data = imgData.data;
      const dpr = window.devicePixelRatio || 1;

      for (let k = 0; k < n; k++) {
        const x = pts[k * 3], y = pts[k * 3 + 1], m = pts[k * 3 + 2];
        const sx = (x - view.cx) * view.scale + W / 2;
        const sy = H / 2 - (y - view.cy) * view.scale;
        const ix = Math.floor(sx * dpr);
        const iy = Math.floor(sy * dpr);
        if (ix < 0 || iy < 0 || ix >= width || iy >= height) continue;
        const idx = (iy * width + ix) * 4;

        let r = 201, g = 212, b = 230;
        if (colorMode === 'map') {
          const hex = CFG.colors[m as keyof typeof CFG.colors] || '#c9d4e6';
          r = parseInt(hex.slice(1, 3), 16);
          g = parseInt(hex.slice(3, 5), 16);
          b = parseInt(hex.slice(5, 7), 16);
        }
        data[idx] = Math.min(255, data[idx] + (r * 0.4 | 0));
        data[idx + 1] = Math.min(255, data[idx + 1] + (g * 0.4 | 0));
        data[idx + 2] = Math.min(255, data[idx + 2] + (b * 0.4 | 0));
        data[idx + 3] = 255;
      }
      ctx.putImageData(imgData, 0, 0);
    }

    // 4. Triangle Level Mode
    if (mode === 'tri' && triData) {
      const tris = triData.triangles;
      const stroke = showStroke || triData.level <= 2;
      const buckets: Record<number, any[]> = { 0: [], 1: [], 2: [], 3: [], 4: [] };
      for (const t of tris) {
        const out = t.outer || 0;
        if (!buckets[out]) buckets[out] = [];
        buckets[out].push(t);
      }

      for (const m of [0, 1, 2, 3, 4]) {
        if (!buckets[m] || buckets[m].length === 0) continue;
        ctx.fillStyle = m === 0 ? '#c9d4e6' : CFG.colors[m as keyof typeof CFG.colors];
        ctx.beginPath();
        for (const t of buckets[m]) {
          const [x0, y0] = worldToScreen(t.v[0][0], t.v[0][1]);
          const [x1, y1] = worldToScreen(t.v[1][0], t.v[1][1]);
          const [x2, y2] = worldToScreen(t.v[2][0], t.v[2][1]);
          ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.lineTo(x2, y2); ctx.closePath();
        }
        ctx.fill('evenodd');
        if (stroke || m === 0) {
          ctx.strokeStyle = 'rgba(10,12,16,0.6)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      if (selectedPiece) {
        const t = tris.find((x: any) => x.a === selectedPiece);
        if (t) {
          ctx.save();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 3]);
          const [x0, y0] = worldToScreen(t.v[0][0], t.v[0][1]);
          const [x1, y1] = worldToScreen(t.v[1][0], t.v[1][1]);
          const [x2, y2] = worldToScreen(t.v[2][0], t.v[2][1]);
          ctx.beginPath();
          ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.lineTo(x2, y2); ctx.closePath();
          ctx.stroke();
          ctx.restore();
        }
      }

      // Draw Labels
      if (showLabels) {
        ctx.font = `${labelSize}px ui-monospace,Menlo,Consolas,monospace`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        const charW = labelSize * 0.62;
        for (const t of tris) {
          const gx = (t.v[0][0] + t.v[1][0] + t.v[2][0]) / 3;
          const gy = (t.v[0][1] + t.v[1][1] + t.v[2][1]) / 3;
          const [sx, sy] = worldToScreen(gx, gy);
          if (sx < -50 || sy < -50 || sx > W + 50 || sy > H + 50) continue;

          const p0 = worldToScreen(t.v[0][0], t.v[0][1]);
          const p1 = worldToScreen(t.v[1][0], t.v[1][1]);
          const p2 = worldToScreen(t.v[2][0], t.v[2][1]);
          const L0 = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
          const L1 = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
          const L2 = Math.hypot(p0[0] - p2[0], p0[1] - p2[1]);
          const Lmax = Math.max(L0, L1, L2);
          const textW = t.a.length * charW + 4;

          if (autoHideLabels && (Lmax < textW * 1.1 || Lmax < 10)) continue;

          ctx.fillStyle = 'rgba(10,12,16,0.65)';
          ctx.fillRect(sx - textW / 2, sy - labelSize / 2 - 1, textW, labelSize + 2);
          ctx.fillStyle = '#ffffff';
          ctx.fillText(t.a, sx, sy);
        }
      }
    }

    // 5. Contact Points U & V
    if (showContact) {
      const fix1: [number, number] = [0, -1];
      const fix2: [number, number] = [2 * alpha / 3, 0];
      const U = MAPS[4](fix1[0], fix1[1], alpha, beta);
      const V = MAPS[2](fix2[0], fix2[1], alpha, beta);
      const [ux, uy] = worldToScreen(U[0], U[1]);
      const [vx, vy] = worldToScreen(V[0], V[1]);
      const dist = Math.hypot(U[0] - V[0], U[1] - V[1]);

      if (dist > 1e-10) {
        ctx.save();
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(ux, uy); ctx.lineTo(vx, vy);
        ctx.stroke();
        ctx.restore();
      }

      ctx.save();
      ctx.fillStyle = '#ff6b6b';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(ux, uy, 6, 0, 2 * Math.PI);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px system-ui';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText('U: 4(23)∞', ux + 10, uy);
      ctx.restore();

      if (dist > 1e-10) {
        ctx.save();
        ctx.fillStyle = '#4dd0e1';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(vx, vy, 6, 0, 2 * Math.PI);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px system-ui';
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText('V: 21∞', vx + 10, vy);
        ctx.restore();
      }
    }
  }, [view, showGrid, showAxes, showOrig, mode, colorMode, showStroke, showLabels, autoHideLabels, labelSize, selectedPiece, showContact, alpha, beta, triData]);

  // Resize listener & Initial draw with ResizeObserver
  useEffect(() => {
    const handleResize = () => {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      if (!container || !canvas) return;
      const rect = container.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      redrawCanvas();
    };

    const timer = setTimeout(handleResize, 10);
    window.addEventListener('resize', handleResize);

    const observer = new ResizeObserver(() => {
      handleResize();
    });
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
      observer.disconnect();
    };
  }, [redrawCanvas]);

  useEffect(() => {
    redrawCanvas();
  }, [redrawCanvas]);

  // Initial draw chaos on mount
  useEffect(() => {
    startChaos(0, 0);
  }, []);

  // --- MOUSE & TOUCH INTERACTION ---
  const isDraggingRef = useRef(false);
  const lastPosRef = useRef({ x: 0, y: 0 });
  const movedRef = useRef(0);

  const handleMouseDown = (e: React.MouseEvent) => {
    isDraggingRef.current = true;
    lastPosRef.current = { x: e.clientX, y: e.clientY };
    movedRef.current = 0;
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingRef.current) return;
    const dx = e.clientX - lastPosRef.current.x;
    const dy = e.clientY - lastPosRef.current.y;
    movedRef.current += Math.abs(dx) + Math.abs(dy);
    lastPosRef.current = { x: e.clientX, y: e.clientY };
    setView(prev => ({
      ...prev,
      cx: prev.cx - dx / prev.scale,
      cy: prev.cy + dy / prev.scale
    }));
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (isDraggingRef.current && movedRef.current < 4 && mode === 'tri' && triData) {
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const ox = e.clientX - rect.left;
        const oy = e.clientY - rect.top;
        const W = rect.width;
        const H = rect.height;
        const wx = (ox - W / 2) / view.scale + view.cx;
        const wy = view.cy - (oy - H / 2) / view.scale;
        
        // Find clicked triangles
        const hits = [];
        for (const t of triData.triangles) {
          if (pointInTri(wx, wy, t.v[0], t.v[1], t.v[2])) hits.push(t);
        }
        if (hits.length > 0) {
          setPieceInfo(hits);
          setSelectedPiece(hits[0].a);
        } else {
          setPieceInfo(null);
          setSelectedPiece(null);
        }
      }
    }
    isDraggingRef.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const ox = e.clientX - rect.left;
    const oy = e.clientY - rect.top;
    const W = rect.width;
    const H = rect.height;
    const wx = (ox - W / 2) / view.scale + view.cx;
    const wy = view.cy - (oy - H / 2) / view.scale;

    const factor = Math.exp(-e.deltaY * 0.0015);
    setView(prev => {
      const newScale = prev.scale * factor;
      return {
        scale: newScale,
        cx: wx - (ox - W / 2) / newScale,
        cy: wy + (oy - H / 2) / newScale
      };
    });
  };

  function pointInTri(px: number, py: number, v0: [number, number], v1: [number, number], v2: [number, number]) {
    const s1 = sign(px, py, v0, v1);
    const s2 = sign(px, py, v1, v2);
    const s3 = sign(px, py, v2, v0);
    const hasNeg = (s1 < 0) || (s2 < 0) || (s3 < 0);
    const hasPos = (s1 > 0) || (s2 > 0) || (s3 > 0);
    return !(hasNeg && hasPos);
  }
  function sign(px: number, py: number, a: [number, number], b: [number, number]) {
    return (px - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (py - b[1]);
  }

  const fitView = () => {
    const R = safeRadius(alpha, beta);
    setView({ cx: 0.25, cy: -0.5, scale: 380 });
  };

  const savePng = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `ifs_a${alpha.toFixed(3)}_b${beta.toFixed(3)}_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  return (
    <div className="flex flex-col h-screen bg-[#0f1115] text-[#e7ecf3] select-none font-sans">
      {/* Header */}
      <header className="px-4 py-2.5 bg-[#171a21] border-b border-[#2a3040] flex flex-wrap items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-md">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-wide">Phòng thí nghiệm IFS Affine — Họ hai tham số & Numba</h1>
            <span className="text-[11px] text-[#9aa3b2]">Hệ thống Chaos Game, xấp xỉ tam giác & Tăng tốc Numba Python</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { setMode('chaos'); if (!chaosPointsRef.current) startChaos(); }}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
              mode === 'chaos' ? 'bg-[#2b5ea8] text-white border border-[#3a78cf]' : 'bg-[#1f2430] text-[#9aa3b2] hover:text-white border border-[#2a3040]'
            }`}
          >
            <Layers className="w-3.5 h-3.5" /> Chaos Attractor
          </button>
          <button
            onClick={() => { setMode('tri'); if (!triData) buildTriLevel(level); }}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
              mode === 'tri' ? 'bg-[#2b5ea8] text-white border border-[#3a78cf]' : 'bg-[#1f2430] text-[#9aa3b2] hover:text-white border border-[#2a3040]'
            }`}
          >
            <Compass className="w-3.5 h-3.5" /> Xấp xỉ Tam giác
          </button>
          <button
            onClick={() => setMode('numba')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
              mode === 'numba' ? 'bg-amber-600 text-white border border-amber-500' : 'bg-[#1f2430] text-amber-400 hover:text-amber-300 border border-[#2a3040]'
            }`}
          >
            <Code className="w-3.5 h-3.5" /> Tối ưu Numba Python
          </button>
        </div>
      </header>

      {/* Main Layout */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[360px_1fr] overflow-hidden">
        {/* Sidebar Controls */}
        <aside className="bg-[#171a21] border-r border-[#2a3040] p-3 overflow-y-auto max-h-[calc(100vh-53px)] space-y-3">
          {mode === 'numba' ? (
            <div className="space-y-3">
              <div className="bg-[#1f2430] border border-[#2a3040] rounded-xl p-3.5 space-y-2">
                <h3 className="text-xs font-semibold text-amber-400 flex items-center gap-1.5">
                  <Sparkles className="w-4 h-4" /> Tăng tốc Numba cho Chaos Game
                </h3>
                <p className="text-[11px] text-[#9aa3b2] leading-relaxed">
                  Thuật toán Chaos Game thực hiện hàng triệu vòng lặp tính toán tọa độ điểm tiếp theo. Trong Python thuần (pure Python), vòng lặp <code>for</code> bị giới hạn bởi tốc độ thông dịch. Bằng cách sử dụng thư viện <b>Numba</b> với decorator <code className="text-amber-300">@njit(fastmath=True)</code>, mã nguồn Python được biên dịch thành mã máy tối ưu (Machine Code) tương đương C/C++, tăng tốc <b>50x – 100x</b>!
                </p>
              </div>

              <div className="bg-[#1f2430] border border-[#2a3040] rounded-xl p-3.5 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[#e7ecf3]">Mã nguồn Python + Numba</span>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(NUMBA_PYTHON_CODE);
                      setCopied(true);
                      setTimeout(() => setCopied(false), 2000);
                    }}
                    className="px-2.5 py-1 bg-[#243047] hover:bg-[#2d3c58] text-xs rounded border border-[#2a3040] flex items-center gap-1 text-[#e7ecf3]"
                  >
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    {copied ? 'Đã chép!' : 'Sao chép'}
                  </button>
                </div>
                <pre className="text-[10px] font-mono bg-[#0b0e13] p-2.5 rounded-lg border border-[#2a3040] text-emerald-300 overflow-x-auto max-h-80 leading-relaxed">
                  {NUMBA_PYTHON_CODE}
                </pre>
              </div>

              <div className="bg-[#1f2430] border border-[#2a3040] rounded-xl p-3.5 space-y-2">
                <h3 className="text-xs font-semibold text-blue-400">Ưu điểm vượt trội của Numba:</h3>
                <ul className="text-[11px] text-[#9aa3b2] space-y-1 list-disc list-inside">
                  <li><b>JIT Compilation:</b> Biên dịch trực tiếp khi chạy lần đầu (Just-In-Time).</li>
                  <li><b>Không phụ thuộc C/C++ extension:</b> Viết hoàn toàn bằng Python thuần với NumPy.</li>
                  <li><b>Fastmath & Parallel:</b> Hỗ trợ vector hóa CPU và tính toán song song đa luồng cực nhanh.</li>
                </ul>
              </div>
            </div>
          ) : (
            <>
              {/* Mode Specific Panel */}
              <div className="bg-[#1f2430] border border-[#2a3040] rounded-xl p-3 space-y-2.5">
                <h3 className="text-xs font-semibold text-blue-400">
                  {mode === 'chaos' ? 'Điều khiển Chaos Game' : 'Điều khiển Xấp xỉ Tam giác'}
                </h3>

                {mode === 'chaos' ? (
                  <>
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-[#9aa3b2] min-w-[65px]">Số điểm</label>
                      <input
                        type="range"
                        min="50000"
                        max="1000000"
                        step="50000"
                        value={numPts}
                        onChange={(e) => setNumPts(parseInt(e.target.value))}
                        className="flex-1 accent-blue-500"
                      />
                      <span className="text-xs font-mono w-16 text-right">{(numPts / 1000).toFixed(0)}k</span>
                    </div>

                    <div className="flex items-center gap-2">
                      <label className="text-xs text-[#9aa3b2] min-w-[65px]">Seed</label>
                      <input
                        type="text"
                        value={seedStr}
                        onChange={(e) => setSeedStr(e.target.value)}
                        className="bg-[#0b0e13] border border-[#2a3040] rounded px-2 py-1 text-xs w-28 text-center font-mono"
                      />
                      <button
                        onClick={() => setSeedStr(String((Math.random() * 1e9) | 0))}
                        className="px-2.5 py-1 bg-[#243047] hover:bg-[#2d3c58] text-xs rounded border border-[#2a3040]"
                      >
                        Seed mới
                      </button>
                    </div>

                    <div className="flex items-center gap-4 pt-1">
                      <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                        <input
                          type="radio"
                          name="col"
                          checked={colorMode === 'mono'}
                          onChange={() => setColorMode('mono')}
                          className="accent-blue-500"
                        /> Một màu
                      </label>
                      <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                        <input
                          type="radio"
                          name="col"
                          checked={colorMode === 'map'}
                          onChange={() => setColorMode('map')}
                          className="accent-blue-500"
                        /> Theo ánh xạ fᵢ
                      </label>
                    </div>

                    <div className="flex items-center gap-2 pt-1">
                      <button
                        onClick={() => startChaos()}
                        disabled={isCalculating}
                        className="flex-1 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 shadow"
                      >
                        <Play className="w-3.5 h-3.5" /> Vẽ Attractor
                      </button>
                      <button
                        onClick={cancelChaos}
                        disabled={!isCalculating}
                        className="px-3 py-2 bg-rose-900/60 hover:bg-rose-800/80 disabled:opacity-50 text-rose-200 rounded-lg text-xs font-semibold border border-rose-800"
                      >
                        Hủy
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-[#9aa3b2] min-w-[65px]">Mức n</label>
                      <input
                        type="range"
                        min="0"
                        max="7"
                        step="1"
                        value={level}
                        onChange={(e) => handleSetLevel(parseInt(e.target.value))}
                        className="flex-1 accent-blue-500"
                      />
                      <span className="text-xs font-mono w-8 text-right">{level}</span>
                    </div>

                    <div className="text-xs text-[#9aa3b2]">
                      Số tam giác: <b className="text-white">{triData ? triData.triangles.length.toLocaleString('vi-VN') : 1}</b>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleSetLevel(Math.max(0, level - 1))}
                        className="flex-1 py-1.5 bg-[#243047] hover:bg-[#2d3c58] text-xs rounded border border-[#2a3040]"
                      >
                        ◀ Mức trước
                      </button>
                      <button
                        onClick={() => handleSetLevel(Math.min(7, level + 1))}
                        className="flex-1 py-1.5 bg-[#243047] hover:bg-[#2d3c58] text-xs rounded border border-[#2a3040]"
                      >
                        Mức sau ▶
                      </button>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleSetLevel(7)}
                        className="flex-1 py-1.5 bg-blue-600 hover:bg-blue-500 text-xs rounded font-medium"
                      >
                        Vẽ ngay mức 7 (16,384)
                      </button>
                    </div>

                    <label className="flex items-center gap-2 text-xs pt-1 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showStroke}
                        onChange={(e) => setShowStroke(e.target.checked)}
                        className="accent-blue-500"
                      /> Viền tam giác phân ranh giới
                    </label>
                  </>
                )}
              </div>

              {/* Parameters Alpha, Beta */}
              <div className="bg-[#1f2430] border border-[#2a3040] rounded-xl p-3 space-y-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold text-blue-400">Tham số α, β</h3>
                  <label className="flex items-center gap-1.5 text-[11px] text-emerald-400 cursor-pointer font-medium">
                    <input
                      type="checkbox"
                      checked={linkedMode}
                      onChange={(e) => handleLinkToggle(e.target.checked)}
                      className="accent-emerald-500"
                    /> Liên kết β = −α/3
                  </label>
                </div>

                <div className="flex items-center gap-2">
                  <label className="text-xs text-[#9aa3b2] min-w-[20px]">α</label>
                  <input
                    type="range"
                    min="-1"
                    max="1"
                    step="0.001"
                    value={alpha}
                    onChange={(e) => handleAlphaChange(parseFloat(e.target.value))}
                    className="flex-1 accent-blue-500"
                  />
                  <input
                    type="number"
                    step="0.001"
                    value={alpha.toFixed(3)}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v)) handleAlphaChange(v);
                    }}
                    className="bg-[#0b0e13] border border-[#2a3040] rounded px-2 py-1 text-xs w-16 text-center font-mono"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <label className="text-xs text-[#9aa3b2] min-w-[20px]">β</label>
                  <input
                    type="range"
                    min="-1"
                    max="1"
                    step="0.001"
                    disabled={linkedMode}
                    value={beta}
                    onChange={(e) => handleBetaChange(parseFloat(e.target.value))}
                    className="flex-1 accent-blue-500 disabled:opacity-40"
                  />
                  <input
                    type="number"
                    step="0.001"
                    disabled={linkedMode}
                    value={beta.toFixed(9)}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v)) handleBetaChange(v);
                    }}
                    className="bg-[#0b0e13] border border-[#2a3040] rounded px-2 py-1 text-xs w-24 text-center font-mono disabled:opacity-40"
                  />
                </div>
                {linkedMode && <div className="text-[10px] text-[#9aa3b2] italic">β được tính tự động theo điều kiện tiếp xúc chuẩn</div>}

                <div className="grid grid-cols-2 gap-2 pt-1">
                  <button
                    onClick={() => { setAlpha(0); setBeta(0); markDirty(0, 0); }}
                    className="py-1.5 bg-[#243047] hover:bg-[#2d3c58] text-xs rounded border border-[#2a3040]"
                  >
                    Đặt α=β=0
                  </button>
                  <button
                    onClick={fitView}
                    className="py-1.5 bg-[#243047] hover:bg-[#2d3c58] text-xs rounded border border-[#2a3040]"
                  >
                    Vừa khung
                  </button>
                  <button
                    onClick={savePng}
                    className="py-1.5 bg-[#243047] hover:bg-[#2d3c58] text-xs rounded border border-[#2a3040] flex items-center justify-center gap-1"
                  >
                    <Download className="w-3.5 h-3.5" /> Tải PNG
                  </button>
                  <button
                    onClick={() => { chaosPointsRef.current = null; setTriData(null); redrawCanvas(); }}
                    className="py-1.5 bg-rose-950/50 hover:bg-rose-900/60 text-xs rounded border border-rose-900/60 text-rose-200"
                  >
                    Xóa hình
                  </button>
                </div>

                <div className="flex items-center gap-2 pt-1">
                  <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                    <input
                      type="checkbox"
                      checked={autoUpdate}
                      onChange={(e) => setAutoUpdate(e.target.checked)}
                      className="accent-blue-500"
                    /> Tự cập nhật khi kéo slider
                  </label>
                </div>
                {isDirty && !autoUpdate && (
                  <div className="text-[11px] text-amber-400 bg-amber-950/30 border border-amber-900/50 p-2 rounded">
                    Tham số đã đổi — bấm <b>Vẽ Attractor</b> để cập nhật.
                  </div>
                )}
              </div>

              {/* Display Options */}
              <div className="bg-[#1f2430] border border-[#2a3040] rounded-xl p-3 space-y-2">
                <h3 className="text-xs font-semibold text-blue-400">Hiển thị</h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <label className="flex items-center gap-1.5 opacity-90 text-slate-300">
                    <input type="checkbox" checked={true} disabled className="accent-blue-500 cursor-not-allowed" /> Trục tọa độ (Luôn hiện)
                  </label>
                  <label className="flex items-center gap-1.5 opacity-90 text-slate-300">
                    <input type="checkbox" checked={true} disabled className="accent-blue-500 cursor-not-allowed" /> Lưới tọa độ (Luôn hiện)
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input type="checkbox" checked={showOrig} onChange={(e) => setShowOrig(e.target.checked)} className="accent-blue-500" /> Tam giác OAB
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} className="accent-blue-500" /> Nhãn địa chỉ
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer col-span-2">
                    <input type="checkbox" checked={showContact} onChange={(e) => setShowContact(e.target.checked)} className="accent-blue-500" /> Hiện cặp điểm tiếp xúc U, V
                  </label>
                </div>
              </div>

              {/* Formula & Contact Info */}
              <div className="bg-[#1f2430] border border-[#2a3040] rounded-xl p-3 space-y-2">
                <h3 className="text-xs font-semibold text-blue-400">Ý nghĩa toán học β = −α/3</h3>
                <p className="text-[11px] text-[#9aa3b2] leading-relaxed">
                  Quan hệ <code className="text-amber-300">β = −α/3</code> bảo đảm hai điểm có địa chỉ <b>4(23)∞</b> và <b>21∞</b> trùng nhau (điểm tiếp xúc U và V), tạo điều kiện cho cấu trúc biên tiếp xúc khít mà không chồng lấn.
                </p>
              </div>
            </>
          )}
        </aside>

        {/* Canvas Area */}
        <div
          ref={containerRef}
          className="relative bg-[#0a0c10] overflow-hidden flex items-center justify-center cursor-grab active:cursor-grabbing"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onWheel={handleWheel}
        >
          <canvas ref={canvasRef} className="block w-full h-full" />

          {/* Progress Overlay */}
          {isCalculating && (
            <div className="absolute top-4 left-4 right-4 max-w-sm bg-[#171a21]/95 border border-[#2a3040] rounded-xl p-3 shadow-xl backdrop-blur">
              <div className="flex justify-between text-xs mb-1.5">
                <span className="font-medium text-blue-400">Đang tính toán Chaos Game</span>
                <span className="font-mono text-[#9aa3b2]">{(progress * 100).toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-[#1f2430] rounded-full overflow-hidden border border-[#2a3040]">
                <div className="h-full bg-blue-500 transition-all duration-75" style={{ width: `${progress * 100}%` }} />
              </div>
              <div className="text-[11px] text-[#9aa3b2] mt-1.5">{progressLabel}</div>
            </div>
          )}

          {/* Selected Piece Info Modal */}
          {pieceInfo && pieceInfo.length > 0 && (
            <div className="absolute top-4 right-4 bg-[#171a21]/95 border border-[#2a3040] rounded-xl p-3.5 shadow-xl max-w-xs text-xs space-y-2 backdrop-blur">
              <div className="flex justify-between items-center border-b border-[#2a3040] pb-2">
                <span className="font-semibold text-blue-400">Thông tin mảnh tam giác</span>
                <button onClick={() => { setPieceInfo(null); setSelectedPiece(null); }} className="text-[#9aa3b2] hover:text-white">✕</button>
              </div>
              {pieceInfo.map((t, idx) => (
                <div key={idx} className="space-y-1 font-mono text-[11px]">
                  <div>Địa chỉ: <b className="text-amber-400">{t.a || '∅'}</b></div>
                  <div className="text-[#9aa3b2]">V₀: ({t.v[0][0].toFixed(3)}, {t.v[0][1].toFixed(3)})</div>
                  <div className="text-[#9aa3b2]">V₁: ({t.v[1][0].toFixed(3)}, {t.v[1][1].toFixed(3)})</div>
                  <div className="text-[#9aa3b2]">V₂: ({t.v[2][0].toFixed(3)}, {t.v[2][1].toFixed(3)})</div>
                </div>
              ))}
            </div>
          )}

          {/* Zoom & Fit Floating Buttons */}
          <div className="absolute bottom-4 right-4 flex flex-col gap-1.5">
            <button
              onClick={() => setView(prev => ({ ...prev, scale: prev.scale * 1.3 }))}
              className="w-9 h-9 bg-[#171a21] hover:bg-[#1f2430] border border-[#2a3040] rounded-lg flex items-center justify-center shadow text-white"
              title="Phóng to"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => setView(prev => ({ ...prev, scale: prev.scale / 1.3 }))}
              className="w-9 h-9 bg-[#171a21] hover:bg-[#1f2430] border border-[#2a3040] rounded-lg flex items-center justify-center shadow text-white"
              title="Thu nhỏ"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={fitView}
              className="w-9 h-9 bg-[#171a21] hover:bg-[#1f2430] border border-[#2a3040] rounded-lg flex items-center justify-center shadow text-white"
              title="Vừa khung hình"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>

          {/* Status bar */}
          <div className="absolute bottom-4 left-4 bg-[#171a21]/90 border border-[#2a3040] rounded-lg px-3 py-1.5 text-[11px] text-[#9aa3b2] shadow backdrop-blur">
            α = <b className="text-white">{alpha.toFixed(3)}</b>, β = <b className="text-white">{beta.toFixed(9)}</b>
            {linkedMode && <span className="text-emerald-400 ml-1.5">(liên kết)</span>} · Chế độ: <b className="text-white">{mode === 'chaos' ? 'Chaos Game' : mode === 'tri' ? 'Tam giác' : 'Numba Python'}</b>
          </div>
        </div>
      </main>
    </div>
  );
}
