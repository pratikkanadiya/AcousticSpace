import React, { useState, useRef, useCallback, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";

const IconBase = ({ children, className }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
    strokeLinecap="round" strokeLinejoin="round" className={className}>
    {children}
  </svg>
);

const Upload = ({ className }) => (
  <IconBase className={className}>
    <path d="M12 16V4M12 4l-4 4M12 4l4 4" />
    <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
  </IconBase>
);

const FileAudio = ({ className }) => (
  <IconBase className={className}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <path d="M14 2v6h6" />
    <path d="M9 15v2M12 14v4M15 15v2" />
  </IconBase>
);

const Radio = ({ className }) => (
  <IconBase className={className}>
    <circle cx="12" cy="12" r="2" />
    <path d="M8.5 8.5a5 5 0 0 0 0 7M15.5 8.5a5 5 0 0 1 0 7M5.5 5.5a9 9 0 0 0 0 13M18.5 5.5a9 9 0 0 1 0 13" />
  </IconBase>
);

const AlertTriangle = ({ className }) => (
  <IconBase className={className}>
    <path d="M12 3 2 20h20L12 3z" />
    <path d="M12 10v4M12 17h.01" />
  </IconBase>
);

const CheckCircle2 = ({ className }) => (
  <IconBase className={className}>
    <circle cx="12" cy="12" r="9" />
    <path d="m8.5 12 2.5 2.5 4.5-5" />
  </IconBase>
);

const Clock = ({ className }) => (
  <IconBase className={className}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3.5 2" />
  </IconBase>
);

const Waves = ({ className }) => (
  <IconBase className={className}>
    <path d="M3 9c1.5-2 3.5-2 5 0s3.5 2 5 0 3.5-2 5 0 3.5 2 5 0" />
    <path d="M3 15c1.5-2 3.5-2 5 0s3.5 2 5 0 3.5-2 5 0 3.5 2 5 0" />
  </IconBase>
);

const Wind = ({ className }) => (
  <IconBase className={className}>
    <path d="M3 8h11a3 3 0 1 0-3-3" />
    <path d="M3 16h14a3 3 0 1 1-3 3" />
  </IconBase>
);

const Building2 = ({ className }) => (
  <IconBase className={className}>
    <path d="M4 22V4a1 1 0 0 1 1-1h8a1 1 0 0 1 1 1v18" />
    <path d="M14 9h5a1 1 0 0 1 1 1v12" />
    <path d="M9 6h.01M9 10h.01M9 14h.01M9 18h.01M17 13h.01M17 17h.01" />
  </IconBase>
);

const Gauge = ({ className }) => (
  <IconBase className={className}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 12 16 8" />
    <path d="M12 3v1M3 12h1M20 12h1M6 6l.7.7M17.3 6.7 18 6" />
  </IconBase>
);

const ChevronRight = ({ className }) => (
  <IconBase className={className}>
    <path d="m9 6 6 6-6 6" />
  </IconBase>
);

const Circle = ({ className }) => (
  <svg viewBox="0 0 24 24" className={className}>
    <circle cx="12" cy="12" r="10" fill="currentColor" />
  </svg>
);

const FontImport = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    .font-mono-ac { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
    .font-sans-ac { font-family: 'IBM Plex Sans', ui-sans-serif, sans-serif; }
    @keyframes scan { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
    @keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
    .scan-line { animation: scan 3.2s linear infinite; }
    .pulse-dot { animation: pulse-dot 2s ease-in-out infinite; }
  `}</style>
);

const WAVE_BARS = Array.from({ length: 140 }, (_, i) => {
  const base = Math.sin(i * 0.22) * 0.5 + Math.sin(i * 0.07) * 0.35;
  const jitter = ((i * 37) % 11) / 11 - 0.5;
  return Math.max(0.08, Math.min(1, Math.abs(base + jitter * 0.3) + 0.15));
});

const METRICS = [
  { key: "rt60", label: "RT60", value: "0.62", unit: "s", icon: Building2, flag: true },
  { key: "drr", label: "DRR", value: "4.1", unit: "dB", icon: Waves, flag: true },
  { key: "c50", label: "CLARITY C50", value: "-1.8", unit: "dB", icon: Gauge, flag: false },
  { key: "centroid", label: "ROOM CENTROID", value: "2140", unit: "Hz", icon: Radio, flag: false },
  { key: "breath_count", label: "BREATH EVENTS", value: "3", unit: "", icon: Wind, flag: true },
  { key: "breath_interval", label: "AVG INTERVAL", value: "1.85", unit: "s", icon: Clock, flag: false },
];


function ConfidenceGauge({ score = 91, verdict = "synthetic" }) {
  const angle = -90 + (score / 100) * 180; // -90..90
  const isSynthetic = verdict === "synthetic";
  const accent = isSynthetic ? "#FF6B5B" : "#7CFFB2";

  const cx = 100, cy = 100, r = 78;
  const arcPoint = (deg) => {
    const rad = (deg * Math.PI) / 180;
    return [cx + r * Math.sin(rad), cy - r * Math.cos(rad)];
  };
  const [x1, y1] = arcPoint(-90);
  const [x2, y2] = arcPoint(90);

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 120" className="w-full max-w-[240px]">
        <path
          d={`M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`}
          fill="none" stroke="#1E2A26" strokeWidth="10" strokeLinecap="round"
        />
        <path
          d={`M ${x1} ${y1} A ${r} ${r} 0 0 1 ${arcPoint(angle)[0]} ${arcPoint(angle)[1]}`}
          fill="none" stroke={accent} strokeWidth="10" strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 6px ${accent}90)` }}
        />
        {[-90, -45, 0, 45, 90].map((d) => {
          const [tx, ty] = arcPoint(d);
          return <circle key={d} cx={tx} cy={ty} r="1.6" fill="#6B7A75" />;
        })}
        <g transform={`rotate(${angle} ${cx} ${cy})`}>
          <line x1={cx} y1={cy} x2={cx} y2={cy - r + 14} stroke={accent} strokeWidth="2.5" />
          <circle cx={cx} cy={cy} r="5" fill={accent} />
        </g>
      </svg>
      <div className="-mt-4 text-center font-mono-ac">
        <div className="text-4xl font-semibold" style={{ color: accent }}>{score}%</div>
        <div className="text-[11px] tracking-[0.2em] text-[#6B7A75] mt-1">CONFIDENCE</div>
      </div>
    </div>
  );
}

function ModelPredictionCard({ analysisResult }) {
  if (!analysisResult) {
    return (
      <div className="mt-4 p-4 rounded-xl border border-[#1E2A26] bg-[#0D1412]">
        <div className="text-[10px] tracking-[0.18em] text-[#6B7A75] mb-2">
          MODEL PREDICTION
        </div>

        <div className="text-sm text-[#6B7A75]">
          Waiting for audio analysis...
        </div>
      </div>
    );
  }

  const verdict = analysisResult.verdict;
  const confidence = Number(analysisResult.confidence_pct) || 0;
  const spoofProbability =
    (Number(analysisResult.spoof_probability) || 0) * 100;

  const isSynthetic = verdict === "synthetic";
  const isAuthentic = verdict === "authentic";

  const prediction = isSynthetic
    ? "AI-GENERATED / SPOOF"
    : isAuthentic
      ? "REAL / AUTHENTIC"
      : "UNKNOWN";

  const description = isSynthetic
    ? `The model predicts this audio is AI-generated / synthetic with ${confidence.toFixed(0)}% confidence.`
    : isAuthentic
      ? `The model predicts this audio is real / bonafide with ${confidence.toFixed(0)}% confidence.`
      : "The model could not determine the audio class.";

  return (
    <div
      className="mt-4 p-5 rounded-xl border"
      style={{
        borderColor: isSynthetic ? "#FF6B5B55" : "#7CFFB255",
        backgroundColor: isSynthetic ? "#FF6B5B08" : "#7CFFB208",
      }}
    >

      {/* Header */}
      <div className="flex items-center gap-2 mb-4">

        <div
          className="w-2 h-2 rounded-full"
          style={{
            backgroundColor: isSynthetic ? "#FF6B5B" : "#7CFFB2",
            boxShadow: isSynthetic
              ? "0 0 8px #FF6B5B"
              : "0 0 8px #7CFFB2",
          }}
        />

        <div className="text-[10px] tracking-[0.18em] text-[#6B7A75]">
          MODEL PREDICTION
        </div>

      </div>

      {/* Prediction */}
      <div
        className="text-xl font-semibold tracking-wide mb-2"
        style={{
          color: isSynthetic ? "#FF6B5B" : "#7CFFB2",
        }}
      >
        {prediction}
      </div>

      {/* Explanation */}
      <div className="text-xs leading-5 text-[#B8C5BF]">
        {description}
      </div>

      {/* Probability information */}
      <div className="mt-4 pt-4 border-t border-[#1E2A26]">

        <div className="grid grid-cols-2 gap-3">

          <div>
            <div className="text-[9px] tracking-[0.12em] text-[#6B7A75]">
              CONFIDENCE
            </div>

            <div className="text-sm text-[#E8F0EC] mt-1">
              {confidence.toFixed(1)}%
            </div>
          </div>

          <div>
            <div className="text-[9px] tracking-[0.12em] text-[#6B7A75]">
              SPOOF PROBABILITY
            </div>

            <div className="text-sm text-[#E8F0EC] mt-1">
              {spoofProbability.toFixed(1)}%
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}

const Play = ({ className }) => (
  <svg viewBox="0 0 24 24" className={className}><path d="M8 5v14l11-7z" fill="currentColor" /></svg>
);

const Pause = ({ className }) => (
  <svg viewBox="0 0 24 24" className={className}>
    <rect x="6" y="5" width="4" height="14" fill="currentColor" />
    <rect x="14" y="5" width="4" height="14" fill="currentColor" />
  </svg>
);

function LiveWaveform({ file }) {
  const containerRef = useRef(null);
  const wavesurferRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!file || !containerRef.current) return;

    setIsReady(false);
    setIsPlaying(false);

    const ws = WaveSurfer.create({
      container: containerRef.current,
      height: 160,
      waveColor: "#2a3a34",
      progressColor: "#7CFFB2",
      cursorColor: "#E8F0EC",
      cursorWidth: 1,
      barWidth: 2,
      barGap: 1,
      barRadius: 1,
      normalize: true,
    });
    wavesurferRef.current = ws;

    const objectUrl = URL.createObjectURL(file);
    ws.load(objectUrl);

    ws.on("ready", () => {
      setIsReady(true);
      setDuration(ws.getDuration());
    });
    ws.on("audioprocess", () => setCurrentTime(ws.getCurrentTime()));
    ws.on("seeking", () => setCurrentTime(ws.getCurrentTime()));
    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));

    return () => {
      ws.destroy();
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  const fmtTime = (t) => {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  return (
    <div>
      <div className="relative rounded-sm border border-[#1E2A26] overflow-hidden">
        <div ref={containerRef} />
        {!isReady && (
          <div className="absolute inset-0 flex items-center justify-center font-mono-ac text-xs text-[#6B7A75]">
            Decoding audio…
          </div>
        )}
      </div>

      <div className="flex items-center justify-between mt-2">
        <button
          onClick={() => wavesurferRef.current?.playPause()}
          disabled={!isReady}
          className="flex items-center gap-1.5 font-mono-ac text-xs px-3 py-1.5 rounded-sm bg-[#7CFFB2] text-[#0B0F0E] font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[#8fffc0] transition-colors"
        >
          {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
          {isPlaying ? "PAUSE" : "PLAY"}
        </button>
        <div className="font-mono-ac text-[11px] text-[#6B7A75]">
          {fmtTime(currentTime)} / {fmtTime(duration)}
        </div>
      </div>
    </div>
  );
}

function WaveformPanel({ file }) {
  return (
    <div className="relative bg-[#121816] border border-[#1E2A26] rounded-sm p-5 overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="font-mono-ac text-[11px] tracking-[0.2em] text-[#6B7A75]">
            SPECTRAL / SPATIAL TRACE
          </div>
          <div className="font-mono-ac text-sm text-[#E8F0EC] mt-0.5">
            {file ? file.name : "case_0417_interview.wav (demo)"}
          </div>
        </div>
        <div className={`flex items-center gap-1.5 font-mono-ac text-[11px] ${file ? "text-[#6B7A75]" : "text-[#FF6B5B]"}`}>
          <Circle className={`w-2 h-2 fill-current ${file ? "" : "pulse-dot"}`} />
          {file ? "AWAITING ANALYSIS" : "MISMATCH DETECTED (DEMO)"}
        </div>
      </div>

      {file ? (
        // Real waveform of the uploaded file. Annotation brackets aren't
        // shown here -- those require the room/breath analysis pipeline
        // (preprocess.py -> FastAPI -> model), which this UI isn't wired
        // to yet. This is genuinely just the audio's waveform.
        <LiveWaveform file={file} />
      ) : (
        <>
          <div className="relative h-40 rounded-sm border border-[#1E2A26]" style={{
            backgroundImage:
              "linear-gradient(#1E2A26 1px, transparent 1px), linear-gradient(90deg, #1E2A26 1px, transparent 1px)",
            backgroundSize: "100% 20%, 5% 100%",
          }}>
            <div className="absolute inset-0 overflow-hidden pointer-events-none opacity-30">
              <div className="scan-line h-full w-24" style={{
                background: "linear-gradient(90deg, transparent, #7CFFB2, transparent)",
              }} />
            </div>

            <div className="absolute inset-0 flex items-center gap-[1.5px] px-2">
              {WAVE_BARS.map((h, i) => {
                const inReflectionWindow = i >= 8 && i <= 24;
                const inMismatch = i >= 78 && i <= 96;
                let color = "#7CFFB2";
                if (inReflectionWindow) color = "#E8F0EC";
                if (inMismatch) color = "#FF6B5B";
                return (
                  <div
                    key={i}
                    style={{ height: `${h * 88}%`, backgroundColor: color, opacity: inMismatch ? 0.9 : 0.75 }}
                    className="flex-1 rounded-[1px]"
                  />
                );
              })}
            </div>
          </div>

          <div className="relative h-14 mt-1">
            <AnnotationBracket left="6%" width="17%" color="#E8F0EC" label="Reflection window · 0–50ms" />
            <AnnotationBracket left="55%" width="14%" color="#6B7A75" label="Breath gap · 1.9s" />
            <AnnotationBracket left="76%" width="18%" color="#FF6B5B" label="Vocal / room mismatch" />
          </div>

          <div className="flex justify-between font-mono-ac text-[10px] text-[#6B7A75] mt-1">
            <span>0:00</span><span>0:15</span><span>0:30</span><span>0:45</span><span>1:00</span>
          </div>
        </>
      )}
    </div>
  );
}

function AnnotationBracket({ left, width, color, label }) {
  return (
    <div className="absolute top-0" style={{ left, width }}>
      <div className="h-2.5 border-l-2 border-r-2 border-t-2 rounded-t-sm" style={{ borderColor: color }} />
      <div className="font-mono-ac text-[10px] mt-1 leading-tight" style={{ color }}>{label}</div>
    </div>
  );
}

function MetricCard({ metric }) {
  const Icon = metric.icon;
  return (
    <div className="bg-[#121816] border border-[#1E2A26] rounded-sm p-3.5 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Icon className="w-3.5 h-3.5 text-[#6B7A75]" />
        {metric.flag && (
          <span className="w-1.5 h-1.5 rounded-full bg-[#FF6B5B]" title="Contributes to mismatch score" />
        )}
      </div>
      <div className="font-mono-ac">
        <div className="text-xl text-[#E8F0EC] leading-none">
          {metric.value}
          <span className="text-xs text-[#6B7A75] ml-1">{metric.unit}</span>
        </div>
        <div className="text-[10px] tracking-[0.15em] text-[#6B7A75] mt-1.5">{metric.label}</div>
      </div>
    </div>
  );
}

function UploadPanel({ onFile, onAnalyze, isAnalyzing, selectedFile }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = useCallback((files) => {
    if (files && files[0]) {
      onFile?.(files[0]);
    }
  }, [onFile]);

  return (
    <div className="bg-[#121816] border border-[#1E2A26] rounded-sm p-4">
      <div className="font-mono-ac text-[11px] tracking-[0.2em] text-[#6B7A75] mb-3">
        SUBMIT CLIP FOR ANALYSIS
      </div>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-sm border-2 border-dashed p-6 flex flex-col items-center gap-2 transition-colors ${
          dragOver ? "border-[#7CFFB2] bg-[#7CFFB2]/5" : "border-[#1E2A26] hover:border-[#3a4a44]"
        }`}
      >
        <input
          ref={inputRef} type="file" accept="audio/*" className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        {selectedFile ? (
          <>
            <FileAudio className="w-6 h-6 text-[#7CFFB2]" />
            <div className="font-mono-ac text-xs text-[#E8F0EC] text-center break-all">{selectedFile.name}</div>
            <div className="font-mono-ac text-[10px] text-[#6B7A75]">Click to replace</div>
          </>
        ) : (
          <>
            <Upload className="w-6 h-6 text-[#6B7A75]" />
            <div className="font-sans-ac text-sm text-[#E8F0EC] text-center">
              Drop an audio file, or click to browse
            </div>
            <div className="font-mono-ac text-[10px] text-[#6B7A75]">WAV · FLAC · MP3 — up to 50MB</div>
          </>
        )}
      </div>
      <button
        onClick={onAnalyze}
        className="w-full mt-3 font-mono-ac text-xs tracking-[0.15em] py-2.5 rounded-sm bg-[#7CFFB2] text-[#0B0F0E] font-semibold hover:bg-[#8fffc0] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        disabled={!selectedFile || isAnalyzing}
      >
        {isAnalyzing ? "ANALYZING..." : "RUN ANALYSIS"}
      </button>
    </div>
  );
}

function HistoryPanel({ history }) {
  return (
    <div className="bg-[#121816] border border-[#1E2A26] rounded-sm p-4">

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="font-mono-ac text-[11px] tracking-[0.2em] text-[#6B7A75]">
          RECENT SCANS
        </div>

        <div className="font-mono-ac text-[9px] text-[#3F514A]">
          {history.length} SCANS
        </div>
      </div>

      {/* Empty state */}
      {history.length === 0 ? (
        <div className="py-6 text-center">
          <div className="font-mono-ac text-[10px] text-[#6B7A75]">
            NO SCANS YET
          </div>

          <div className="font-mono-ac text-[9px] text-[#3F514A] mt-1">
            Analysis history will appear here
          </div>
        </div>
      ) : (
        <div className="space-y-1.5">

          {history.map((item, index) => {

            const confidence = Number(item.confidence) || 0;

            return (
              <div
                key={`${item.filename}-${index}`}
                className="flex items-center justify-between gap-2 px-2.5 py-2 rounded-sm border border-[#1E2A26] bg-[#0E1311] hover:border-[#2D3C35] transition-colors"
              >

                {/* File information */}
                <div className="flex items-center gap-2 min-w-0">

                  <FileAudio className="w-3.5 h-3.5 text-[#6B7A75] shrink-0" />

                  <div className="min-w-0">
                    <div
                      className="font-mono-ac text-[10px] text-[#B8C5BF] truncate"
                      title={item.filename}
                    >
                      {item.filename}
                    </div>
                  </div>

                </div>

                {/* Confidence */}
                <div className="shrink-0 text-right">

                  <div className="font-mono-ac text-[9px] text-[#6B7A75]">
                    CONF.
                  </div>

                  <div className="font-mono-ac text-[11px] text-[#7CFFB2]">
                    {confidence.toFixed(1)}%
                  </div>

                </div>

              </div>
            );
          })}

        </div>
      )}
    </div>
  );
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [history, setHistory] = useState(() => {
    try {
      const savedHistory = localStorage.getItem("acousticspace_history");

      return savedHistory
        ? JSON.parse(savedHistory)
        : [];
    } catch (error) {
      console.error("Failed to load history:", error);
      return [];
    }
  });
  const [serverStatus, setServerStatus] = useState("checking"); // 'checking', 'ready', 'error'
  const [errorMsg, setErrorMsg] = useState(null);

  // Check server health on load
  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then(res => res.json())
      .then(data => {
        if (data.status === "ok") setServerStatus("ready");
        else setServerStatus("error");
      })
      .catch(() => setServerStatus("error"));
  }, []);

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    setErrorMsg(null);
    
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();

      setAnalysisResult(data);

      /*
      * Save scan to history
      */
      const newHistoryItem = {
        filename: data.filename || selectedFile.name,
        confidence: Number(data.confidence_pct) || 0,
      };

      setHistory((previousHistory) => {

        const updatedHistory = [
          newHistoryItem,
          ...previousHistory,
        ];

        // Keep only latest 10 scans
        const limitedHistory = updatedHistory.slice(0, 10);

        // Persist history
        localStorage.setItem(
          "acousticspace_history",
          JSON.stringify(limitedHistory)
        );

        return limitedHistory;
      });
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Map API data to metrics array, fallback to zeroed data if no result yet
  const displayMetrics = analysisResult ? [
    { key: "rt60", label: "RT60", value: analysisResult.acoustic_summary.rt60_sec.toFixed(2), unit: "s", icon: Building2, flag: true },
    { key: "drr", label: "DRR", value: analysisResult.acoustic_summary.drr_db.toFixed(1), unit: "dB", icon: Waves, flag: true },
    { key: "c50", label: "CLARITY C50", value: analysisResult.acoustic_summary.room_clarity_c50.toFixed(1), unit: "dB", icon: Gauge, flag: false },
    { key: "centroid", label: "ROOM CENTROID", value: Math.round(analysisResult.acoustic_summary.room_centroid), unit: "Hz", icon: Radio, flag: false },
    { key: "breath_count", label: "BREATH EVENTS", value: Math.round(analysisResult.acoustic_summary.breath_count), unit: "", icon: Wind, flag: true },
    { key: "breath_interval", label: "AVG INTERVAL", value: analysisResult.acoustic_summary.avg_breath_interval.toFixed(2), unit: "s", icon: Clock, flag: false },
  ] : [
    { key: "rt60", label: "RT60", value: "--", unit: "s", icon: Building2, flag: false },
    { key: "drr", label: "DRR", value: "--", unit: "dB", icon: Waves, flag: false },
    { key: "c50", label: "CLARITY C50", value: "--", unit: "dB", icon: Gauge, flag: false },
    { key: "centroid", label: "ROOM CENTROID", value: "--", unit: "Hz", icon: Radio, flag: false },
    { key: "breath_count", label: "BREATH EVENTS", value: "--", unit: "", icon: Wind, flag: false },
    { key: "breath_interval", label: "AVG INTERVAL", value: "--", unit: "s", icon: Clock, flag: false },
  ];

  return (
    <div className="min-h-screen bg-[#0B0F0E] font-sans-ac">
      <FontImport />

      {/* top bar */}
      <div className="border-b border-[#1E2A26] px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-sm bg-[#7CFFB2]/10 border border-[#7CFFB2]/30 flex items-center justify-center">
            <Radio className="w-4 h-4 text-[#7CFFB2]" />
          </div>
          <div>
            <div className="font-mono-ac text-sm font-semibold text-[#E8F0EC] tracking-[0.08em]">
              ACOUSTICSPACE
            </div>
            <div className="font-mono-ac text-[10px] text-[#6B7A75] tracking-[0.15em]">
              INFOTACT · ANALYST CONSOLE
            </div>
          </div>
        </div>
        <div className={`flex items-center gap-2 font-mono-ac text-[11px] ${serverStatus === 'ready' ? 'text-[#7CFFB2]' : 'text-[#FF6B5B]'}`}>
          <Circle className={`w-2 h-2 fill-current ${serverStatus === 'ready' ? 'pulse-dot' : ''}`} />
          {serverStatus === 'checking' && "CHECKING SYSTEM..."}
          {serverStatus === 'ready' && "SYSTEM READY"}
          {serverStatus === 'error' && "API OFFLINE"}
        </div>
      </div>

      {/* body */}
      <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr_260px] gap-4 p-5 max-w-[1400px] mx-auto">
        {/* left rail */}
        <div className="flex flex-col gap-4">
          <UploadPanel 
            selectedFile={selectedFile}
            onFile={(f) => { setSelectedFile(f); setAnalysisResult(null); setErrorMsg(null); }} 
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
          />
          {errorMsg && (
            <div className="bg-[#FF6B5B]/10 border border-[#FF6B5B]/30 rounded-sm p-3 font-mono-ac text-xs text-[#FF6B5B]">
              {errorMsg}
            </div>
          )}
          <HistoryPanel history={history} />
        </div>

        {/* main */}
        <div className="flex flex-col gap-4">
          <WaveformPanel file={selectedFile} />
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {displayMetrics.map((m) => <MetricCard key={m.key} metric={m} />)}
          </div>
        </div>

        {/* right rail */}
        <div className="flex flex-col gap-4">
          <div className="bg-[#121816] border border-[#1E2A26] rounded-sm p-4">
            <div className="font-mono-ac text-[11px] tracking-[0.2em] text-[#6B7A75] mb-1">
              VERDICT
            </div>
            <ConfidenceGauge 
              score={analysisResult ? Math.round(analysisResult.confidence_pct) : 0} 
              verdict={analysisResult ? analysisResult.verdict : "unknown"} 
            />
            <ModelPredictionCard analysisResult={analysisResult} />
            <div className="font-sans-ac text-xs text-[#6B7A75] text-center mt-2 leading-relaxed">
              {analysisResult 
                ? (analysisResult.verdict === "synthetic" 
                    ? "Vocal cadence implies a small acoustically-treated room; measured reflections don't match."
                    : "Room acoustics and breath patterns align with natural physical environments.")
                : "Awaiting analysis..."}
            </div>
          </div>

          <div className="bg-[#121816] border border-[#1E2A26] rounded-sm p-4">
            <div className="font-mono-ac text-[11px] tracking-[0.2em] text-[#6B7A75] mb-2">
              CLIP INFO
            </div>
            <dl className="font-mono-ac text-xs space-y-1.5">
              {[
                ["DURATION", analysisResult ? `${analysisResult.duration_sec.toFixed(1)}s` : "--"],
                ["SAMPLE RATE", analysisResult ? `${(analysisResult.sample_rate / 1000).toFixed(1)} kHz` : "--"],
                ["SEGMENTS", analysisResult ? analysisResult.num_segments : "--"],
                ["CLIPPING", analysisResult ? (analysisResult.is_clipped ? "Detected" : "None") : "--"],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between text-[#6B7A75]">
                  <dt>{k}</dt>
                  <dd className={v !== "--" ? "text-[#E8F0EC]" : ""}>{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </div>
    </div>
  );

  
}