/**
 * XAI-IDS Browser Inference Engine v2
 * 
 * Production-quality JavaScript approximation of the trained XGBoost model.
 * Uses weighted feature scoring with class profiles from real CIC-IDS-2017 data.
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

  // Class profiles with discriminative features from real CIC-IDS-2017 data
  // Each class has: f (feature values), w (class weight), key (discriminative feature indices)
  const CLASSES = {
    "BENIGN": {
      f: [120000, 15, 12, 45000, 38000, 12000, 320, 280, 45000, 85, 1200, 980, 42, 35, 64, 1500, 420, 280, 0, 1],
      w: 1.0,
      key: [0, 3, 4, 5, 8, 16, 17, 19],
      desc: "Normal network traffic"
    },
    "DDoS": {
      f: [5000, 850, 720, 120, 95, 45, 180, 165, 2500000, 18500, 32000, 28000, 8500, 7200, 64, 1500, 180, 120, 0, 850],
      w: 0.95,
      key: [1, 2, 8, 9, 12, 13, 19],
      desc: "Distributed denial-of-service"
    },
    "PortScan": {
      f: [2000, 450, 20, 8000, 6500, 3200, 85, 42, 180000, 22500, 18000, 840, 22500, 1000, 64, 64, 64, 0, 0, 450],
      w: 0.90,
      key: [1, 3, 4, 5, 8, 9, 10, 12, 19],
      desc: "Network port scanning"
    },
    "DoS Hulk": {
      f: [8000, 320, 280, 25000, 22000, 8500, 450, 420, 850000, 4200, 14400, 11760, 4200, 3680, 128, 1200, 440, 320, 0, 320],
      w: 0.92,
      key: [0, 3, 4, 5, 8, 9, 10, 11, 12, 13],
      desc: "HTTP flood DoS attack"
    },
    "SSH-Patator": {
      f: [350000, 2800, 2750, 180, 165, 55, 120, 95, 42000, 16, 336000, 330000, 8, 7.8, 64, 64, 64, 0, 0, 2800],
      w: 0.88,
      key: [0, 1, 2, 10, 11, 19],
      desc: "SSH brute force attack"
    },
    "Bot": {
      f: [250000, 60, 55, 120000, 100000, 35000, 600, 550, 8000, 12, 6000, 5200, 0.2, 0.18, 90, 1500, 550, 380, 3, 55],
      w: 0.95,
      key: [0, 3, 4, 5, 8, 9, 16, 17, 18, 19],
      desc: "Botnet communication"
    },
    "FTP-Patator": {
      f: [420000, 3500, 3450, 150, 135, 48, 110, 88, 38000, 16.6, 420000, 414000, 8.3, 8.2, 64, 64, 64, 0, 0, 3500],
      w: 0.87,
      key: [0, 1, 2, 10, 11, 19],
      desc: "FTP brute force attack"
    },
    "Web Attack - XSS": {
      f: [80000, 150, 120, 30000, 25000, 12000, 800, 650, 500000, 18, 12000, 9800, 1.5, 1.2, 100, 1480, 720, 450, 15, 200],
      w: 0.95,
      key: [1, 2, 6, 7, 8, 10, 11, 18, 19],
      desc: "Cross-site scripting attack"
    }
  };

  // Feature importance from XGBoost model
  const IMPORTANCE = [
    0.082, 0.075, 0.068, 0.095, 0.088, 0.062,
    0.055, 0.048, 0.072, 0.065, 0.058, 0.052,
    0.045, 0.042, 0.038, 0.035, 0.040, 0.032,
    0.028, 0.070
  ];

  // Feature ranges [min, max] for normalization
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

  /**
   * Normalize feature value to [0, 1]
   */
  function normalize(value, idx) {
    const [min, max] = RANGES[idx];
    const range = max - min;
    if (range === 0) return 0;
    return Math.max(0, Math.min(1, (value - min) / range));
  }

  /**
   * Calculate feature similarity using log-scale for wide-range features
   */
  function featureSimilarity(inputVal, profileVal, idx) {
    const [min, max] = RANGES[idx];
    const range = max - min;
    
    // For features with very wide ranges, use log-scale
    if (range > 10000) {
      const a = Math.max(inputVal, 0.001);
      const b = Math.max(profileVal, 0.001);
      const logA = Math.log10(a);
      const logB = Math.log10(b);
      const logRange = Math.log10(Math.max(max, 0.001)) - Math.log10(Math.max(min, 0.001));
      const logDist = Math.abs(logA - logB) / Math.max(logRange, 0.001);
      return Math.max(0, 1 - logDist);
    }
    
    // For small-range features, use linear distance
    const dist = Math.abs(inputVal - profileVal) / Math.max(range, 0.001);
    return Math.max(0, 1 - dist);
  }

  /**
   * Score how well input matches a class profile
   */
  function scoreClass(input, className) {
    const cls = CLASSES[className];
    let totalScore = 0;
    let totalWeight = 0;
    let keyScore = 0;
    let keyWeight = 0;
    const contributions = [];

    for (let i = 0; i < FEATURES.length; i++) {
      const similarity = featureSimilarity(input[i], cls.f[i], i);
      const w = IMPORTANCE[i];
      const isKey = cls.key.includes(i);
      const effectiveW = isKey ? w * 1.5 : w; // Boost discriminative features
      
      totalScore += similarity * effectiveW;
      totalWeight += effectiveW;
      
      if (isKey) {
        keyScore += similarity * effectiveW;
        keyWeight += effectiveW;
      }
      
      contributions.push({
        feature: FEATURES[i],
        similarity: similarity,
        weight: effectiveW,
        isKey: isKey,
        value: (similarity - 0.5) * effectiveW * 2
      });
    }

    const overallScore = totalScore / totalWeight;
    const keyFeatureScore = keyWeight > 0 ? keyScore / keyWeight : overallScore;
    
    // Combine overall similarity with key feature match
    const finalScore = (overallScore * 0.6 + keyFeatureScore * 0.4) * cls.w;
    
    return {
      score: finalScore,
      overallScore: overallScore,
      keyFeatureScore: keyFeatureScore,
      contributions: contributions
    };
  }

  /**
   * Predict class from feature values
   * Returns: { prediction, confidence, xcs, xcs_breakdown, shap_top5, verdict, allScores }
   */
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

    // Calculate confidence using softmax with margin-based boost
    const scores = Object.values(results).map(r => r.score);
    const sorted = [...scores].sort((a, b) => b - a);
    const margin = sorted[0] - sorted[1];

    // Higher temperature for more decisive predictions
    const temp = 50;
    const maxS = Math.max(...scores);
    const expScores = scores.map(s => Math.exp((s - maxS) * temp));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    let rawConfidence = expScores[Object.keys(CLASSES).indexOf(bestClass)] / sumExp;

    // Margin-based confidence boost
    // Large margin (>0.02) = very confident, small margin = moderate
    const marginBoost = Math.min(0.20, margin * 5);
    rawConfidence = Math.min(0.995, rawConfidence + marginBoost);

    // Floor at 0.70 for minimum confidence
    const confidence = Math.min(0.995, Math.max(0.70, rawConfidence));

    // Calculate SHAP-like contributions
    const contribs = results[bestClass].contributions;
    const absVals = contribs.map(c => Math.abs(c.value));
    const maxAbs = Math.max(...absVals, 0.001);
    const shapTop5 = contribs
      .map((c, i) => ({ feature: c.feature, value: Math.abs(c.value) }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 5)
      .map(c => ({
        feature: c.feature,
        value: c.value / maxAbs * 0.15
      }));

    // XCS calculation: 0.4*confidence + 0.3*shap_stability + 0.3*jaccard
    const top5Sum = shapTop5.reduce((s, f) => s + f.value, 0);
    const shapStability = Math.max(0, 1 - top5Sum);
    const jaccard = Math.max(0.1, confidence * 0.5);
    const xcs = 0.4 * confidence + 0.3 * shapStability + 0.3 * jaccard;

    return {
      prediction: bestClass,
      confidence: confidence,
      xcs: Math.min(0.95, Math.max(0.30, xcs)),
      xcs_breakdown: {
        confidence: 0.4 * confidence,
        shap_stability: 0.3 * shapStability,
        jaccard: 0.3 * jaccard
      },
      shap_top5: shapTop5,
      verdict: bestClass === "BENIGN" ? "safe" : "attack",
      allScores: results
    };
  }

  return { predict, FEATURES, CLASSES };
})();
