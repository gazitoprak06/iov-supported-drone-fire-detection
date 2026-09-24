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

# Bound up front so that a missing artifact prints a message instead of raising
# a NameError further down. A reader who has not run the pipeline yet is in
# exactly that state, and this is the script that is supposed to tell them so.
rel = {}
rf = {}

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
    print("\nV2 metrikleri bulunamadi. (v2_learned_baselines.py calistirildi mi?)")

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

    # The null baselines for this cohort are the audit the paper is named for,
    # so the summary states them next to the result rather than leaving the
    # reader to assume the result clears them.
    nb_path = os.path.join("Proje_Kodlari", "evaluation_results", "v3_deep_edge",
                           "v3_null_baselines.json")
    if os.path.exists(nb_path):
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f).get("unseen_by_gradient", {})
        print("  (Null sabit-pozitif F1 : {:.2f}%)".format(
            nb.get("constant_positive", {}).get("F1", 0)))
        print("  (Null sabit-negatif Acc: {:.2f}%  <-> V3 Acc: {:.2f}%)".format(
            nb.get("constant_negative", {}).get("Accuracy", 0),
            nb.get("deep_edge", {}).get("Accuracy", 0)))
else:
    print("\nV3 metrikleri bulunamadi.")

# All three architectures on one cohort. This is the comparison Table 5 could
# not make until the deep edge weights were applied to the still-image split.
st_path = os.path.join("Proje_Kodlari", "evaluation_results", "v3_deep_edge",
                       "v3_on_still_images.json")
if os.path.exists(st_path) and rel and rf:
    with open(st_path, "r", encoding="utf-8") as f:
        st = json.load(f).get("deep_edge_still", {})
    print("\n--- ORTAK KUME: ucu de ayni 410 goruntude ---")
    print("(V3 bu goruntulerin hicbirinde egitilmedi: alan aktarimi)")
    print("V1 kural tabanli          : F1 = {:.2f}%   Acc = {:.2f}%".format(
        rel.get("F1", 0), rel.get("Accuracy", 0)))
    print("V3 derin ag (aktarim)     : F1 = {:.2f}%   Acc = {:.2f}%".format(
        st.get("F1", 0), st.get("Accuracy", 0)))
    print("V2 ogrenilmis (bu veride egitildi): F1 = {:.2f}%   Acc = {:.2f}%".format(
        rf.get("F1", 0), rf.get("Accuracy", 0)))
    if st.get("F1") is not None and rel.get("F1") is not None:
        if st["F1"] > rel["F1"]:
            print("  -> Kendi verisi icin ayarlanmis kural, o veriyi hic gormemis aga yeniliyor.")
        else:
            print("  -> Bu kohortta aktarim, kural tabanli sistemi gecemiyor.")

print("\n" + "="*70)
print("  Bu rakamlarin tamami yayimlanan sonuc dosyalarindan okundu.")
print("  Dogrulamak icin:  python tools\\check_paper_numbers.py")
print("="*70)
