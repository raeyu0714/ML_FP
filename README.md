# XFeat × RoMa：影像配對 baseline、研究結果與 Hybrid 改善

> I2ML-2026 Investigation Track 專題 README  
> 本文件依照「先理解影像配對，再比較兩個 baseline，最後分析改善」的順序撰寫。互動深度與粗到細定位的數據來自已完成的 pilot；兩個 baseline 的公平重跑與最後 Hybrid 結果仍需依同一協定完成。

## 1. 什麼是影像配對模型

影像配對（image matching）是給定兩張不同視角的影像，找出代表同一個三維場景位置的像素對應：

```text
Image A: pixel p_A  ─────────┐
                             ├─ same 3D point
Image B: pixel p_B  ─────────┘
```

一個配對模型通常包含：

1. **特徵提取**：把 RGB 影像轉成能描述局部結構的 feature map 或 descriptor。
2. **對應搜尋**：在另一張影像中找可能的對應位置。
3. **匹配判定與定位**：計算相似度、confidence，並輸出像素座標。
4. **幾何驗證**：把配對交給 RANSAC/PoseLib 等姿態 solver，估計相機相對姿態。

配對模型的結果會用在：

- 相機姿態估計
- Structure-from-Motion 與 3D 重建
- 視覺定位與地圖建立
- 影像拼接與跨視角搜尋

我們把錯誤分成兩種：

- **搜尋錯誤**：正確位置根本沒有被搜尋到。
- **定位錯誤**：已經在正確區域附近，但座標誤差仍超過像素門檻。

![XFeat 配對示意圖](assets/xfeat_paper_matching.png)

*圖 1：XFeat 配對示意圖。綠色線段表示兩張影像中被判定為對應的局部特徵，可交給姿態估計或 3D 重建。圖像取自 [XFeat, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.pdf)，不是本專案的測量結果。*


## 2. 目標模型與兩個 baseline

本專案的兩個目標模型：

- **XFeat baseline**：代表輕量、稀疏候選、低延遲的配對路徑。
- **RoMa baseline**：代表全局 coarse matching、連續座標和 coarse-to-fine refinement 的配對路徑。

兩者先作為共同比較起點。只有在完成相同資料、相同解析度、相同姿態 solver 和相同計時邊界的 baseline 評估後，才開始加入新模組。

| 比較項目 | XFeat baseline | RoMa baseline |
|---|---|---|
| 特徵 | 輕量共享 CNN、多尺度特徵、64D descriptor | 粗尺度 foundation features 加專用 fine features |
| 搜尋介面 | detector 選出的稀疏候選點 | 粗特徵圖上的全局位置分布 |
| 座標輸出 | 從候選位置中選擇 | 連續 coarse 座標，再做局部細化 |
| 配對輸出 | sparse pairs，可做 mutual nearest filtering | dense / semi-dense correspondence 與 confidence |
| baseline 目的 | 測量效率與候選限制 | 測量完整搜尋與定位能力 |

## 3. 兩個 baseline 的架構與優缺點

### XFeat baseline

![XFeat baseline architecture](assets/xfeat_architecture_ppt.svg)

*圖 3：依照提供的 G2L_XFeat_architecture 簡報整理。x3 約 1/8、x4 約 1/16、x5 約 1/32；融合後產生 keypoint/reliability 與 64D descriptor。*

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

**優點**

- CNN 和 descriptor 輕量，CPU 推論速度快。
- 只比較稀疏候選，記憶體與相似度計算量低。
- 適合低延遲或資源有限的部署。

**限制**

- 真實位置若不在候選集合，後端無法輸出集合外的座標。
- 重複紋理或弱紋理可能讓 detector 漏掉幾何有效位置。
- 候選覆蓋、descriptor 選擇和座標精度混在同一條稀疏介面，失敗來源不容易單獨修正。

### RoMa baseline

![RoMa baseline architecture](assets/roma_architecture_paper.svg)

*圖 4：依 RoMa 論文 Figure 2 重畫。粗特徵、細特徵、Transformer match decoder、warp refiner 和兩種 supervision 分開表示。*

![RoMa 論文架構原圖](assets/roma_paper_architecture.jpg)

*圖 5：RoMa 論文 Figure 2 原圖裁切，說明 coarse encoder、fine encoder、match decoder 和 loss 的角色。*

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

**優點**

- 不必先把真值限制在 detector 候選點中。
- 可以先搜尋全局粗位置，再輸出連續座標。
- coarse 與 fine 階段分工，能分別處理大範圍搜尋和局部定位。

**限制**

- 全局相關性和 dense/sami-dense 輸出需要較多計算、記憶體和資料搬移。
- coarse stage 若已經跑到錯誤區域，fine refinement 未必能救回來。
- 完整模型的速度和準確率不能只用參數量推算，必須在相同硬體實測。

## 相關性與連續座標公式

RoMa-style coarse matching 先對兩張影像的粗特徵建立位置相似度。對 A 圖位置 i 和 B 圖位置 j：

$$
C(i,j)=\frac{F_A(i)^\top F_B(j)}{\sqrt{d}}
$$

把同一個 A 圖位置對 B 圖所有位置的相似度轉成權重：

$$
s_{ij}=\frac{\exp(C(i,j))}{\sum_k\exp(C(i,k))}
$$

再用權重讀出 B 圖的連續座標：

$$
\hat p_B(i)=\sum_j s_{ij}p_B(j)
$$

這些公式只說明「相似度 → 位置分布 → 連續座標期望」；完整 RoMa baseline 還包含特徵編碼、match decoder、confidence 和 warp refinement。

## 4. 資料集、切割與訓練預算

### 為什麼使用 MegaDepth

MegaDepth 適合當作共同資料，因為：

- **多視角場景**：同一建築或場景有不同視角，能測大視角、遮擋和尺度變化。
- **幾何標註**：SfM/MVS 提供深度、相機和可見性，可計算像素級真值。
- **場景多樣**：網路照片涵蓋不同光線、相機、紋理和拍攝距離。
- **可比較兩個 baseline**：同一組 RGB、深度和相機條件能支援稀疏與密集配對。
- **可做 scene-disjoint 驗證**：驗證場景與訓練場景分開，降低記住建築外觀的風險。

MegaDepth 的限制是 MVS 深度可能有噪聲、空洞和不可重建區域；資料下載、解壓和前處理也很大。因此每次訓練要保存 manifest、有效 pair 數、深度完整性和資料 hash。

![MegaDepth 論文資料示例](assets/megadepth_paper_figure.png)

*圖 6：MegaDepth 論文 Figure 1 裁切，展示網路影像、深度預測與跨資料集泛化概念。來源：[MegaDepth, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.pdf)。此圖是資料說明，不代表本專案已下載論文中的全部原始檔。*

### 訓練與驗證切割

| 部分 | 規劃 |
|---|---|
| 訓練 split | MegaDepth 官方訓練場景，依 scene 固定 manifest |
| validation split | 與訓練場景不重疊的 464 對 dev 影像 |
| 最終測試 split | 保留未參與選型的場景，模型選定後才開封 |
| 每對資料 | RGB、深度、相機參數、可見性與 pair metadata |
| 固定條件 | 輸入解析度、pair sampling、姿態 solver、solver seeds |

### 預計訓練曝光數

- 每個可訓練分支目標為 **2,000,000 次 image-pair exposure**。
- batch size 為 **8 對影像**；若每一步固定取 8 對，約等於 **250,000 optimizer steps**。
- exposure 是累計看到的影像對，不是 2,000,000 組互不重複影像，也不是 2,000,000 steps。
- XFeat 和 RoMa baseline 先以官方權重做 inference；需要訓練的改善分支再使用相同 exposure 預算。
- 每個分支保存 model、optimizer、scheduler/scaler、step、data progress、loss log 和 config，確保可中斷恢復。
- 若有效 pair 數不足，要記錄實際曝光和重複取樣規則，不能把縮減資料版寫成完整重現。

## 5. 研究結果

### 5.1 Baseline 首輪結果：尚未測量

兩個 baseline 的公平結果目前還沒有填入。完成同一 validation split、同一 solver seeds 和同一計時邊界後，才更新下表。

| 指標 | XFeat baseline | RoMa baseline | 評估規則 |
|---|---:|---:|---|
| Pose AUC@5 | 未測量 | 未測量 | 0°–5° 姿態誤差曲線面積 |
| Pose AUC@10 | 未測量 | 未測量 | 0°–10° 姿態誤差曲線面積 |
| Pose AUC@20 | 未測量 | 未測量 | 0°–20° 姿態誤差曲線面積 |
| correct@3 | 未測量 | 未測量 | 配對像素誤差小於 3 px |
| correct@5 | 未測量 | 未測量 | 配對像素誤差小於 5 px |
| candidate / dense coverage | 未測量 | 未測量 | 幾何有效真值是否被搜尋介面涵蓋 |
| CPU 平均 s/pair | 未測量 | 未測量 | 1 CPU、0 GPU、不含姿態 solver |
| CPU p50 / p95 s/pair | 未測量 | 未測量 | 相同 warm-up 和計時範圍 |
| GPU latency | 未測量 | 未測量 | 固定硬體、batch 和同步方式 |
| 峰值 GPU 記憶體 | 未測量 | 未測量 | 推論階段記錄 |
| 參數量 | 未測量 | 未測量 | 以實際 checkpoint 計算 |

### 5.2 AUC@5 的解釋

AUC@5 是姿態誤差曲線在 0° 到 5° 範圍內的 area under curve，通常正規化成百分比，越高越好。

![AUC@5 示意](assets/auc5_explainer.svg)

*圖 7：橫軸是估計姿態誤差，縱軸是誤差不超過該值的影像對比例。AUC@5 累積 0–5° 的整段曲線，不是單一 5° 通過率。*

AUC、correct@3 和 coverage 是不同層級的指標，不能互換：

- AUC 評估配對交給姿態 solver 後的整體幾何結果。
- correct@3 / correct@5 評估單一配對點的像素誤差。
- coverage 評估搜尋介面是否包含幾何有效真值位置。

## 6. 已有研究結果：互動深度階梯

這些數據是目前已完成的 pilot，可作為後續研究的依據。它們不是兩個 baseline 的公平重跑，也不代表 Hybrid 已經完成。

| 分支 | Frozen AUC@5 | Unfrozen AUC@5 | Frozen CPU s/pair | Unfrozen CPU s/pair |
|---|---:|---:|---:|---:|
| I0：基本匹配頭 | 27.82% | 20.00% | 1.139 | 1.448 |
| I1：一層圖內上下文 | 32.55% | 37.42% | 1.469 | 1.678 |
| I2：一層圖內加跨圖互動 | 35.33% | 38.15% | 1.947 | 2.231 |
| I4：兩次重複互動 | 36.63% | 42.72% | 2.795 | 3.041 |
| I6：六次重複互動 | 46.85% | 49.10% | 6.128 | 6.390 |

![互動深度 AUC](assets/focus_int_auc.png)

![互動深度 CPU](assets/focus_int_cpu.png)

解讀：在這個 pilot 中，互動深度增加通常伴隨 AUC 增加與 CPU 成本增加。下一個決策不是直接選 I6，而是確認較淺分支的增益能否在共同 baseline 與未見 scene 重現。

## 7. 已有研究結果：粗到細定位階梯

這組 pilot 先把全局座標讀出、粗修正和細修正拆開：

| 分支 | Frozen AUC@5 | Unfrozen AUC@5 | Frozen CPU s/pair | Unfrozen CPU s/pair |
|---|---:|---:|---:|---:|
| L0：global correlation + coordinate expectation | 5.59% | 7.92% | 1.507 | 1.525 |
| L1：L0 + coarse refinement | 5.27% | 15.65% | 1.981 | 1.992 |
| L2：L1 + fine refinement | 7.39% | 16.56% | 2.127 | 2.135 |

![粗細定位 AUC](assets/focus_stage_auc.png)

![粗細定位 CPU](assets/focus_stage_cpu.png)

解讀：在這個 pilot 的 unfrozen 條件下，L0→L1 增加 7.73 個百分點，L1→L2 增加 0.91 個百分點。這支持優先檢查 coarse stage，但不代表 fine stage 沒有作用。

## 8. 最後的 Hybrid：結合上下文與粗到細定位

Hybrid 是在兩條階梯都完成後才測的結合模型，目的不是直接把所有東西疊上去，而是檢查搜尋改善和定位改善是否互補。

```text
XFeat shared CNN
  → optional context block (I1 or selected depth)
  → global correlation (L0)
  → coarse refinement (L1)
  → fine refinement (L2)
  → final correspondence
  → fixed PoseLib evaluation
```

### Hybrid 消融矩陣

| 組合 | 上下文互動 | 全局／座標細化 | 目前狀態 |
|---|---|---|---|
| baseline | 無 | 無 | 未測量的共同起點 |
| context-only | 選定的 I1 或最佳淺層 | 無 | 未測量 |
| refinement-only | 無 | L0→L1→L2 | 未測量 |
| full hybrid | 選定互動深度 | L0→L1→L2 | 未測量 |

Hybrid 必須與兩個 baseline 使用相同 validation split、exposure、輸入解析度、姿態 solver 和時間邊界。至少要同時回報 AUC、correct、coverage、CPU/GPU latency 和峰值記憶體，才能判斷改善是否值得。

## 9. 研究流程與停止規則

1. 固定 XFeat 與 RoMa baseline 的版本、權重、輸入解析度、manifest 和計時邊界。
2. 完成 baseline 首輪比較，填入第 5.1 節的「未測量」欄位。
3. 固定資料有效性：scene split、pair hash、深度完整性、相機幾何和有效 pair 數。
4. 依序分析 I0→I1→I2→I4→I6。
5. 依序分析 L0→L1→L2。
6. 只選通過成本和重現門檻的模組進入 Hybrid。
7. 對搜尋失敗、粗階段錯配、附近定位失敗和姿態 solver 失敗分開統計。
8. 只有在多個 scene 可重現、增益高於測量誤差且成本可接受時，才進入長訓和最終測試。

每個實驗開始前寫清楚：它要改變的架構決策、證據缺口、曝光預算、pass/stop 條件。實驗結束後更新方法、數據、限制和下一步。

## 10. 課程分工與交付

| 角色 | 工作 | 交付 |
|---|---|---|
| 資料與評估 | manifest、geometry sanity、AUC/correct、split | evaluation script、protocol |
| XFeat baseline | backbone、候選、descriptor、CPU profile | baseline report |
| RoMa baseline | coarse/fine pipeline、速度與記憶體 | baseline report |
| 改善模組 | I0–I6、L0–L2、Frozen/Unfrozen、Hybrid | model code、tables |
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

## 目前狀態

已完成：

- 影像配對問題、XFeat/RoMa baseline 架構與優缺點說明。
- MegaDepth 理由、資料切割與 2,000,000 exposure 預算。
- 互動深度和粗到細定位 pilot 數據。
- Hybrid 的架構與消融矩陣定義。

尚未完成：

- 兩個 baseline 在共同資料與硬體條件下的首輪公平數值。
- Hybrid 的正式訓練與評估。
- 未見場景測試、多 seed 不確定度和完整 GPU/CPU/記憶體報告。

## 可重現原則

每筆結果至少要能回答：

- 使用哪個 commit、config、manifest 和 scene split？
- 更新幾步、看過多少 image-pair exposure？
- AUC 的資料、solver seed 和聚合方式？
- 計時是否包含讀圖、前處理、特徵、匹配和姿態？
- checkpoint 能否恢復，結果能否由原始 log 重算？

若答案不完整，標為 partial 或 diagnostic，不標為正式模型結論。
