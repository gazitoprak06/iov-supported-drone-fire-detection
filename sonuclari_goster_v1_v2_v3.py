import json
import os

# Paths are built with os.path.join and resolved against this file rather than
# against the working directory of the caller, so the summary runs unchanged on
# any platform and from any directory.
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

print("="*70)
print("  DRONE YANGIN TESPIT SISTEMI - BASARI ORANLARI (V1, V2, V3)")
print("="*70)

# V2 JSON file
v2_path = os.path.join("Proje_Kodlari", "evaluation_results", "v2_baselines",
                       "v2_baseline_metrics.json")
if os.path.exists(v2_path):
    with open(v2_path, "r", encoding="utf-8") as f:
        v2_data = json.load(f)
    print("\n--- V1: KURAL TABANLI SISTEM (Hand-set Heuristic) ---")
    print("Makale Tablo 1 sonuclarina gore Test Seti uzerinde:")
    # V1 figures are read from the image-level artifact rather than printed as
    # literals, so that this summary cannot drift away from the experiment.
    v1_path = os.path.join("Proje_Kodlari", "evaluation_results", "v1_image_level",
                           "v1_image_level_metrics.json")
    if os.path.exists(v1_path):
        with open(v1_path, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
        rel = v1_data.get("released_configuration", {})
        print("Dogruluk (Accuracy)     : {:.2f}%".format(rel.get("Accuracy", 0)))
        print("Dogru Bilme (Recall/TPR): {:.2f}%".format(rel.get("Recall", 0)))
        print("Yanlis Alarm (FPR)      : {:.2f}%".format(rel.get("FPR", 0)))
        print("F1-Skoru                : {:.2f}%".format(rel.get("F1", 0)))
        nulls = v1_data.get("null_baselines", {})
        print("  (Null sabit-pozitif F1 : {:.2f}%)".format(
            nulls.get("constant_positive", {}).get("F1", 0)))
        print("  (Null sabit-negatif Acc: {:.2f}%)".format(
            nulls.get("constant_negative", {}).get("Accuracy", 0)))
    else:
        print("V1 metrikleri bulunamadi. (tools/eval_image_level.py calistirildi mi?)")
    print("Gecikme (Latency): {:.2f} ms/frame".format(v2_data.get("latency", {}).get("hand_set_pipeline", {}).get("median_ms", 0)))
    
    print("\n--- V2: MAKINE OGRENMESI (Learned Baselines) ---")
    rf = v2_data.get("learned", {}).get("rf_color_bands", {})
    lr = v2_data.get("learned", {}).get("lr_color_bands", {})
    print("Lojistik Regresyon (Tum Ozellikler): F1 = {:.2f}%".format(lr.get("F1", 0)))
    print("Random Forest (Tum Ozellikler)   : F1 = {:.2f}%".format(rf.get("F1", 0)))
else:
    print("\nV2 metrikleri bulunamadi. (train_v2_baselines.py calistirildi mi?)")

# V3 JSON file
v3_path = os.path.join("Proje_Kodlari", "evaluation_results", "v3_deep_edge",
                       "v3_results_by_split.json")
if os.path.exists(v3_path):
    with open(v3_path, "r", encoding="utf-8") as f:
        v3_data = json.load(f)
    print("\n--- V3: DERIN OGRENME (MobileNetV3) ---")
    print("483 Kliplik Gercek Drone Videolari Uzerinde:")
    test_group = v3_data.get("unseen_by_gradient", {})
    if test_group:
        print("Egitimde Gorulmeyen (Unseen) {} Klipte Basari:".format(test_group.get("N_clips", 0)))
        print("Dogru Bilme (Recall/TPR): {:.2f}%".format(test_group.get("Recall_TPR", 0)))
        print("Yanlis Alarm (FPR)      : {:.2f}%".format(test_group.get("FPR", 0)))
        print("F1-Skoru                : {:.2f}%".format(test_group.get("F1", 0)))
else:
    print("\nV3 metrikleri bulunamadi.")

print("\n" + "="*70)
