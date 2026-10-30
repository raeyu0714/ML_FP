# XFeat × RoMa：兩個 baseline 的影像配對比較與改善計畫

> I2ML-2026 Investigation Track 專題 README  
> 本專案先把 XFeat 與 RoMa 都當作 baseline，先在同一資料與評估協定下量測差異，再決定哪些模組值得加入 XFeat。README 中「研究改善」目前全部是待執行計畫；尚未把任何改善分支的結果當成已完成成果。

## 研究目標

給定兩張不同視角的場景影像，我們要找出代表同一個三維位置的像素對應，並把這些對應交給幾何估計器恢復相機相對姿態。

研究分成三層：

1. **Baseline 比較**：公平比較 XFeat 的效率路徑與 RoMa 的全局粗到細路徑。
2. **失敗分析**：判斷錯誤來自候選搜尋、粗定位、細定位或姿態 solver。
3. **待測改善**：保留 XFeat backbone，逐步加入上下文、全局相關性與座標細化，量化準確率和成本是否值得。

![XFeat 論文的 sparse / semi-dense 配對](assets/xfeat_paper_matching.png)

*圖 1：XFeat 論文 Figure 2。上排是 sparse matching，下排是 semi-dense matching；綠色線段是兩張影像的對應，可交給姿態估計和 3D 重建。來源：[XFeat, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.pdf)。*

![RoMa 論文的 dense matching 示例](assets/roma_paper_matching.png)

*圖 2：RoMa 論文 Figure 1。不同視角、光線和紋理條件下的密集對應視覺化。來源：[RoMa, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.pdf)。*

## 兩個 baseline 的架構與優缺點

| 比較項目 | XFeat baseline | RoMa baseline |
|---|---|---|
| 特徵 | 輕量共享 CNN、多尺度特徵、64D descriptor | 粗尺度 foundation features 加專用 fine features |
| 搜尋介面 | detector 選出的稀疏候選點 | 粗特徵圖上的全局位置分布 |
| 座標輸出 | 只能從候選位置選擇 | 連續 coarse 座標，再做局部細化 |
| 配對輸出 | sparse pairs，可做 mutual nearest filtering | dense / semi-dense correspondence 與 confidence |
| 主要優勢 | CPU 速度快、記憶體低、容易部署 | 搜尋範圍完整、適合大視角和連續座標 |
| 主要限制 | 真值不在候選集合時，後端無法補出 | 全局搜尋、解碼和 refinement 成本較高 |
| 在本專案的角色 | 效率 baseline | 準確率與完整搜尋 baseline |

### Baseline 首輪比較：目前尚未測量

兩個 baseline 必須使用同一批影像、同一輸入解析度、同一姿態 solver 和同一計時邊界。以下表格先保留結果欄位，完成公平重跑後再填入數字。

| 指標 | XFeat baseline | RoMa baseline | 狀態與規則 |
|---|---:|---:|---|
| Pose AUC@5 | 未測量 | 未測量 | 同一 validation split、同一 solver seeds |
| Pose AUC@10 | 未測量 | 未測量 | 同上 |
| Pose AUC@20 | 未測量 | 未測量 | 同上 |
| correct@3 | 未測量 | 未測量 | 像素誤差小於 3 px 的配對比例 |
| correct@5 | 未測量 | 未測量 | 像素誤差小於 5 px 的配對比例 |
| candidate / correspondence coverage | 未測量 | 未測量 | 分別記錄可搜尋位置覆蓋 |
| CPU 前端平均 s/pair | 未測量 | 未測量 | 1 CPU、0 GPU，不含姿態 solver |
| CPU p50 / p95 s/pair | 未測量 | 未測量 | 相同 warm-up 和計時範圍 |
| GPU latency | 未測量 | 未測量 | 記錄硬體、batch 和同步方式 |
| 峰值 GPU 記憶體 | 未測量 | 未測量 | 推論階段單獨記錄 |
| 參數量 | 未測量 | 未測量 | 以實際 checkpoint 和設定計算 |

## XFeat baseline 架構

這張圖依照提供的 G2L_XFeat_architecture 簡報重新整理，把 backbone、三個尺度、skip fusion、descriptor 和配對介面分開。

![XFeat baseline architecture](assets/xfeat_architecture_ppt.svg)

*圖 3：XFeat baseline。x3 約 1/8、x4 約 1/16、x5 約 1/32；融合後產生 keypoint/reliability 與 64D descriptor，最後只在有限候選點上做相似度配對。*

流程：

```text
Image A / Image B
  → shared XFeat CNN
  → x3 (1/8), x4 (1/16), x5 (1/32)
  → s = x3 + up(x4), deep feature d = x5
  → keypoint / reliability + 64D descriptor
  → sparse candidate points
  → descriptor similarity + mutual nearest filtering
  → matched pairs → pose / 3D geometry
```

瓶頸是候選介面：

```text
真實連續位置：       ×
已選候選位置：          ●     ●
```

若真值沒有進入候選集合，即使 descriptor 已辨認出正確物體，模型也只能輸出候選集合中的近似位置。

## RoMa baseline 架構

RoMa 改變的是搜尋介面：先在粗尺度對整張特徵圖建模，再把 coarse 結果交給 fine refinement。

![RoMa baseline architecture](assets/roma_architecture_paper.svg)

*圖 4：依 RoMa 論文 Figure 2 重畫的 baseline 流程。粗特徵、細特徵、Transformer match decoder、warp refiner 和兩種 supervision 分開表示。*

![RoMa 論文架構原圖](assets/roma_paper_architecture.jpg)

*圖 5：RoMa 論文 Figure 2 原圖裁切。這張圖用來解釋 coarse encoder、fine encoder、match decoder 和 loss 的角色。*

流程：

```text
Image A / Image B
  → frozen foundation coarse features
  → global coarse matching distribution
  → anchor probabilities / continuous coarse coordinate
  → specialized fine features
  → warp refinement
  → dense correspondence + confidence
  → pose / geometry evaluation
```

RoMa 可以在固定候選點之外讀出連續座標，因此搜尋較完整；代價是全局相關性、解碼和 refinement 都需要更多計算。

## 全局相關性與連續座標

對 A 圖位置 i 和 B 圖位置 j 的粗特徵，相關性是：

$$
C(i,j)=\frac{F_A(i)^\top F_B(j)}{\sqrt{d}}
$$

令每個 B 圖位置的權重為：

$$
s_{ij}=\frac{\exp(C(i,j))}{\sum_k\exp(C(i,k))}
$$

則初始連續座標是：

$$
\hat p_B(i)=\sum_j s_{ij}p_B(j)
$$

這些公式只表達「相似度 → 位置分布 → 連續座標期望」；完整 baseline 還包含特徵編碼、解碼、confidence 和 refinement。

## 資料集、切割與訓練預算

### 為什麼使用 MegaDepth

MegaDepth 適合作為這個研究的共同資料，原因是：

- **多視角場景**：同一建築或場景有不同視角，能直接測試大視角與遮擋。
- **幾何標註**：SfM/MVS 提供深度、相機和可見性，可計算像素級幾何真值。
- **場景多樣**：來自網路照片，包含不同光線、尺度、相機和紋理。
- **可支援兩個 baseline**：稀疏配對和密集 coarse-to-fine 配對都能使用同一組影像與幾何條件。
- **可做 scene-disjoint 驗證**：把驗證場景與訓練場景分開，避免只記住某些建築外觀。

MegaDepth 也有明確限制：MVS 深度可能有噪聲、空洞和不可重建區域；下載、解壓與資料前處理成本高。因此所有結果都要記錄有效 pair 數、深度完整性和實際 manifest，不能只用資料集名稱宣稱公平。

![MegaDepth 論文資料示例](assets/megadepth_paper_figure.png)

*圖 6：MegaDepth 論文 Figure 1 裁切，展示網路影像、深度預測與跨資料集泛化概念。來源：[MegaDepth, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.pdf)。此圖是資料說明，不代表本專案已下載論文中的全部原始檔。*

### 本專案預計切割

| 部分 | 規劃 |
|---|---|
| 訓練 split | MegaDepth 官方訓練場景；以 scene 為單位固定 manifest |
| 驗證 split | 與訓練場景不重疊的 464 對 dev 影像 |
| 最終測試 | 保留未參與選型的場景，待 baseline 與模組選定後才開封 |
| 幾何資料 | RGB、深度、相機參數、可見性與 pair metadata |
| 固定條件 | 輸入解析度、pair sampling、姿態 solver、solver seeds |

### 預計訓練曝光數

- 每個可訓練分支目標為 **2,000,000 次 image-pair exposure**。
- batch size 為 **8 對影像**；若每一步固定取 8 對，對應約 **250,000 optimizer steps**。
- exposure 是累計看到的影像對，不是 2,000,000 組互不重複影像，也不是 2,000,000 steps。
- XFeat 和 RoMa 的官方權重先做 inference baseline；需要訓練的改善分支再依相同 exposure 預算訓練。
- 每個分支保存 model、optimizer、scheduler/scaler、step、data progress、loss log 和 config，確保可中斷恢復。
- 若資料有效 pair 數不足，必須在 manifest 中記錄實際曝光和重複取樣規則，不得把不足資料寫成完整重現。

## AUC@5 是什麼

AUC@5 是姿態誤差曲線在 0° 到 5° 範圍內的 area under curve，通常正規化成百分比，越高越好。

![AUC@5 示意](assets/auc5_explainer.svg)

*圖 7：橫軸是估計姿態誤差，縱軸是誤差不超過該值的影像對比例。AUC@5 累積 0–5° 的整段曲線，不是單一 5° 通過率。*

本專案同時記錄：

- **AUC@5 / @10 / @20**：影像對經過固定姿態 solver 後的曲線面積。
- **correct@3 / correct@5**：單一配對點的像素誤差是否低於門檻。
- **coverage**：候選或 dense correspondence 是否包含幾何有效的真值位置。
- **CPU/GPU time**：分開記錄讀圖、前處理、特徵、匹配、姿態與同步邊界。

AUC、correct@3 和 coverage 是不同層級的指標，不能互換。

## 待執行的研究改善

以下模組全部標為「未測量」。它們不是目前已完成的結果，而是 baseline 公平比較完成後才啟動的研究階段。

### 互動深度階梯

| 代號 | 計畫加入的模組 | 預期回答的問題 | AUC/CPU 狀態 |
|---|---|---|---|
| I0 | descriptor projection + basic matching head | 沒有上下文的共同控制 | 未測量 |
| I1 | 一層圖內 self-attention | 少量上下文是否改善粗搜尋 | 未測量 |
| I2 | 圖內互動後加一層跨圖互動 | 跨圖訊息是否額外有用 | 未測量 |
| I4 | 重複兩次圖內／跨圖互動 | 深度增加是否持續有效 | 未測量 |
| I6 | 重複六次圖內／跨圖互動 | 準確率增益是否值得延遲 | 未測量 |

### 粗到細定位階梯

| 代號 | 計畫加入的模組 | 預期回答的問題 | AUC/CPU 狀態 |
|---|---|---|---|
| L0 | global correlation + coordinate expectation | 全局座標讀出作為共同控制 | 未測量 |
| L1 | L0 + coarse coordinate refinement | 粗修正能救回多少初始誤差 | 未測量 |
| L2 | L1 + fine coordinate refinement | 細尺度還能增加多少定位精度 | 未測量 |

Frozen 與 Unfrozen 各自測量，形成 16 個計畫 cell：

- 互動階梯：I0、I1、I2、I4、I6 × Frozen/Unfrozen。
- 定位階梯：L0、L1、L2 × Frozen/Unfrozen。
- L0-Frozen 若沒有可訓練定位頭，只作 forward-only control，不標成成功訓練。

## 研究流程與決策規則

1. 固定 XFeat 和 RoMa baseline 的版本、權重、輸入解析度、資料 manifest 和計時邊界。
2. 在相同 464 對 validation split 上完成第一張 baseline 比較表；目前所有數值先保持「未測量」。
3. 檢查資料有效性：scene split、pair hash、深度完整性、相機幾何和有效 pair 數。
4. 先跑 I0→I1→I2→I4→I6，找出最低成本的有效上下文深度。
5. 再跑 L0→L1→L2，分開判斷全局搜尋、粗定位和細定位。
6. 組合四個消融：baseline、context-only、refinement-only、full hybrid。
7. 對每個 cell 保存 AUC、correct、coverage、CPU/GPU 時間、記憶體、loss、checkpoint 和 config。
8. 只有在多個 scene 可重現、增益高於測量誤差且成本仍可接受時，才進入長訓和未見場景測試。

每個實驗開始前要寫清楚貢獻、證據缺口、訓練曝光預算、pass/stop 條件；結束後更新方法、數據、限制和下一步。

## 目前狀態

已完成：

- XFeat 與 RoMa baseline 的架構、優缺點和公平比較規則。
- XFeat backbone、RoMa coarse-to-fine 架構圖。
- 論文配對示例、AUC@5 說明圖和 MegaDepth 資料示例。
- 待測的 I0–I6、L0–L2 模組定義。

尚未完成：

- XFeat / RoMa 在同一資料和硬體條件下的 baseline 數值；表格目前全部是「未測量」。
- 16 個改善 cell 的訓練與評估。
- 四組 hybrid 消融、未見場景測試和多 seed 不確定度。
- 完整 GPU/CPU/記憶體報告。

## 課程分工與交付

| 角色 | 工作 | 交付 |
|---|---|---|
| 資料與評估 | manifest、geometry sanity、AUC/correct、split | evaluation script、protocol |
| XFeat baseline | backbone、候選、descriptor、CPU profile | baseline report |
| RoMa baseline | coarse/fine pipeline、速度與記憶體 | baseline report |
| 改善模組 | I0–I6、L0–L2、Frozen/Unfrozen | model code、tables |
| 分析與簡報 | 圖表、限制、proposal、個人貢獻 | figures、slides、contribution log |

## Repository 結構

```text
ML_FP/
├── README.md
├── assets/
├── src/
├── scripts/
├── configs/
├── experiments/
├── figures/
└── requirements.txt
```

不要提交影像、深度、私密資料、API key、虛擬環境或大型 checkpoint。大型結果只記錄外部路徑、hash、step 和取得方式。

## 可重現原則

每筆結果至少要能回答：

- 使用哪個 commit、config、manifest 和 scene split？
- 更新幾步、看過多少 image-pair exposure？
- AUC 的資料、solver seed 和聚合方式？
- 計時是否包含讀圖、前處理、特徵、匹配和姿態？
- checkpoint 能否恢復，結果能否由原始 log 重算？

若答案不完整，標為 partial 或 diagnostic，不標為正式模型結論。
