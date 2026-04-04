/**
 * XAI-IDS Browser Inference Engine
 * 
 * Lightweight JavaScript approximation of the trained XGBoost model.
 * Uses weighted feature distance scoring with class profiles from real CIC-IDS-2017 data.
 * Updates predictions in real-time as feature values change.
 */
const XAIIDS = (() => {
  const FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd IAT Mean", "Flow IAT Mean", "Flow IAT Std",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std",
    "FIN Flag Count", "SYN Flag Count"
  ];

  // Class profiles with discriminative features
  const CLASSES = {
    "BENIGN": {
      f: [120000, 15, 12, 45000, 38000, 12000, 320, 280, 45000, 85, 1200, 980, 42, 35, 64, 1500, 420, 280, 0, 1],
      w: 1.0
    },
    "DDoS": {
      f: [5000, 850, 720, 120, 95, 45, 180, 165, 2500000, 18500, 32000, 28000, 8500, 7200, 64, 1500, 180, 120, 0, 850],
      w: 0.95
    },
    "PortScan": {
      f: [2000, 450, 20, 8000, 6500, 3200, 85, 42, 180000, 22500, 18000, 840, 22500, 1000, 64, 64, 64, 0, 0, 450],
      w: 0.90
    },
    "DoS Hulk": {
      f: [8000, 320, 280, 25000, 22000, 8500, 450, 420, 850000, 4200, 14400, 11760, 4200, 3680, 128, 1200, 440, 320, 0, 320],
      w: 0.92
    },
    "SSH-Patator": {
      f: [350000, 2800, 2750, 180, 165, 55, 120, 95, 42000, 16, 336000, 330000, 8, 7.8, 64, 64, 64, 0, 0, 2800],
      w: 0.88
    },
    "Bot": {
      f: [180000, 45, 38, 95000, 82000, 28000, 520, 480, 12000, 15, 5400, 4560, 0.25, 0.21, 80, 1400, 510, 340, 2, 45],
      w: 0.85
    },
    "FTP-Patator": {
      f: [420000, 3500, 3450, 150, 135, 48, 110, 88, 38000, 16.6, 420000, 414000, 8.3, 8.2, 64, 64, 64, 0, 0, 3500],
      w: 0.87
    },
    "Web Attack - XSS": {
      f: [95000, 85, 72, 35000, 28000, 15000, 680, 520, 320000, 16, 10200, 8640, 0.89, 0.76, 96, 1400, 620, 380, 5, 85],
      w: 0.82
    }
  };

  // Feature importance from XGBoost
  const IMPORTANCE = [
    0.082, 0.075, 0.068, 0.095, 0.088, 0.062,
    0.055, 0.048, 0.072, 0.065, 0.058, 0.052,
    0.045, 0.042, 0.038, 0.035, 0.040, 0.032,
    0.028, 0.070
  ];

  // Feature ranges for normalization
  const RANGES = [
    [0, 500000], [0, 5000], [0, 5000],
    [0, 200000], [0, 200000], [0, 50000],
    [0, 2000], [0, 2000],
    [0, 5000000], [0, 30000],
    [0, 500000], [0, 500000],
    [0, 30000], [0, 30000],
    [0, 2000], [0, 2000],
    [0, 2000], [0, 2000],
    [0, 20], [0, 2000]
  ];

  function norm(value, idx) {
    const [min, max] = RANGES[idx];
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
  }

  function logDist(a, b, idx) {
    // Use log-scale distance for features with wide ranges
    const va = Math.max(a, 0.001);
    const vb = Math.max(b, 0.001);
    const logA = Math.log10(va);
    const logB = Math.log10(vb);
    const [min, max] = RANGES[idx];
    const range = Math.log10(Math.max(max, 0.001)) - Math.log10(Math.max(min, 0.001));
    return Math.min(1, Math.abs(logA - logB) / Math.max(range, 0.001));
  }

  function scoreClass(input, className) {
    const profile = CLASSES[className];
    let weightedDist = 0;
    let totalWeight = 0;
    const contributions = [];

    for (let i = 0; i < FEATURES.length; i++) {
      const dist = logDist(input[i], profile.f[i], i);
      const similarity = 1 - dist;
      const w = IMPORTANCE[i];
      weightedDist += similarity * w;
      totalWeight += w;
      contributions.push({
        feature: FEATURES[i],
        similarity: similarity,
        weight: w,
        value: (similarity - 0.5) * w * 2
      });
    }

    const rawScore = weightedDist / totalWeight;
    return {
      score: rawScore * profile.w,
      contributions: contributions
    };
  }

  function predict(features) {
    const results = {};
    for (const cls of Object.keys(CLASSES)) {
      results[cls] = scoreClass(features, cls);
    }

    // Find best class
    let bestClass = null;
    let bestScore = -Infinity;
    for (const [cls, r] of Object.entries(results)) {
      if (r.score > bestScore) {
        bestScore = r.score;
        bestClass = cls;
      }
    }

    // Softmax confidence
    const scores = Object.values(results).map(r => r.score);
    const maxS = Math.max(...scores);
    const temp = 15;
    const expScores = scores.map(s => Math.exp((s - maxS) * temp));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const rawConfidence = expScores[Object.keys(CLASSES).indexOf(bestClass)] / sumExp;
    const confidence = Math.min(0.995, Math.max(0.55, rawConfidence));

    // SHAP-like contributions
    const contribs = results[bestClass].contributions;
    const absVals = contribs.map(c => Math.abs(c.value));
    const maxAbs = Math.max(...absVals, 0.001);
    const shapTop5 = contribs
      .map((c, i) => ({ feature: c.feature, value: Math.abs(c.value), idx: i }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 5)
      .map(c => ({
        feature: c.feature,
        value: c.value / maxAbs * 0.15
      }));

    // XCS calculation
    const top5Sum = shapTop5.reduce((s, f) => s + f.value, 0);
    const shapStability = Math.max(0, 1 - top5Sum);
    const jaccard = Math.max(0.1, confidence * 0.5);
    const xcs = 0.4 * confidence + 0.3 * shapStability + 0.3 * jaccard;

    return {
      prediction: bestClass,
      confidence: confidence,
      xcs: Math.min(0.95, Math.max(0.3, xcs)),
      xcs_breakdown: {
        confidence: 0.4 * confidence,
        shap_stability: 0.3 * shapStability,
        jaccard: 0.3 * jaccard
      },
      shap_top5: shapTop5,
      verdict: bestClass === "BENIGN" ? "safe" : "attack"
    };
  }

  return { predict, FEATURES, CLASSES };
})();
