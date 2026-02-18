# ===========================
# Configuration
# ===========================

$datasets = @(
    #@{ path = "data/nam_similar_v_associated.xlsx"; prefix = "nam" },
    @{ path = "data/nam_func_v_perc.xlsx";                 prefix = "ldt" }
)

$models = @(
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased",
    "microsoft/mpnet-base",
    "Aunsiels/ChildBERT",
    "albert-base-v2",
    "FacebookAI/xlm-roberta-base",
    "bert-large-uncased"
)

# Components to extract
# Assumes extractor.py will skip invalid layers per model
$components = @(
    "word_embeddings",
    "encoder_layer_1",
    "encoder_layer_2",
    "encoder_layer_3",
    "encoder_layer_4",
    "encoder_layer_5",
    "encoder_layer_6",
    "encoder_layer_7",
    "encoder_layer_8",
    "encoder_layer_9",
    "encoder_layer_10",
    "encoder_layer_11",
    "encoder_layer_12"
)

foreach ($dataset in $datasets) {
    foreach ($model in $models) {
        foreach ($component in $components) {

            Write-Host "Running: $model | $($dataset.prefix) | $component" -ForegroundColor Cyan

            python extractor.py `
                "$($dataset.path)" `
                -c "$component" `
                -i "$model" `
                -p "$($dataset.prefix)" `
                --rhos "D:/func_v_perc/rhos.json" `
                --cache "D:/func_v_perc" `
                --bs 500
        }
    }
}