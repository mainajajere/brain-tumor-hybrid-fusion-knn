#!/bin/bash
echo "=== XAI DEBUG ==="

echo "1. Test CSV exists?"
[ -f results/splits/test.csv ] && echo "YES ($(wc -l < results/splits/test.csv) lines)" || echo "NO"

echo "2. Aux models saved?"
[ -d results/models/aux_logits ] && echo "YES" || echo "NO"
[ -d results/models/aux_cam ] && echo "YES" || echo "NO"

echo "3. Test images exist?"
head -n 3 results/splits/test.csv | while read line; do
  path=$(echo "$line" | cut -d',' -f1)
  [ -f "$path" ] && echo "YES: $path" || echo "MISSING: $path"
done

echo "4. Grad-CAM output?"
ls results/xai/gradcam/*.png 2>/dev/null | wc -l

echo "5. SHAP output?"
ls results/xai/shap/*.png 2>/dev/null | wc -l
