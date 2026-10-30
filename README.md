# 從稀疏特徵到全局定位：輕量雙視圖影像配對的拆解研究

> 課程專題（Investigation Track）工作說明
>
> 這份 README 是給組員的共同起點：先把問題、假設、分支、評估方式與分工說清楚，再開始實作。所有結果都必須以可重現的資料、設定與紀錄為準；目前的數字是先前拆解實驗的內部 pilot，不能直接當成新方法已經成功。

## 30 秒版本

給定兩張從不同角度拍攝的照片，我們希望找到代表同一個場景位置的對應點，並用這些對應估計兩張照片的相機相對姿態。

本專案不先假設哪一種現成系統最好，而是把輕量雙視圖配對流程拆成兩條可控制的軸：

1. **互動深度軸**：從沒有互動、圖內上下文、一次跨圖互動，到多次重複互動。
2. **定位階段軸**：從全局相關性，到粗座標修正，再到粗細兩階段修正。

每個分支都做 frozen／unfrozen 對照，回答「是哪一個模組帶來改善」以及「改善需要多少計算成本」。最後再把有效模組組成一個混合模型，檢查準確率與效率是否能取得更好的折衷。

## 課程對應

本專案採用 [I2ML-2026 Investigation Track](https://nthu-cs-hmilab.github.io/I2ML-2026/project/)：

- 找出既有影像配對流程的限制與失敗模式。
- 用控制變因的實驗分析「何時失敗、為什麼失敗」。
- 提出可驗證的模組化修正，並以共同資料與共同指標比較。
- 最後提供結論、限制、可重現程式與每位組員的實際貢獻。

課程目前公布的時程：

| 事項 | 日期 |
|---|---|
| 3 分鐘 proposal video | 11/03（Tue） |
| 期末簡報 | 12/17、12/22、12/24 |
| Final report／code | 以課程後續公告為準 |

## 問題定義

影像配對可以拆成四步：

```text
兩張影像
  → 特徵提取
  → 搜尋可能的對應位置
  → 判斷配對與座標
  → 幾何姿態估計
```

我們要分開處理兩種錯誤：

- **搜尋錯誤**：真實對應沒有進入模型正在搜尋的區域。
- **定位錯誤**：已經找到正確區域，但最後座標不夠精確，沒有通過像素誤差門檻。

這兩種錯誤需要不同的模組。上下文互動偏向改善搜尋；座標修正偏向改善定位。若把所有模組一次加入，最後即使分數改變，也無法知道是哪個原因造成。

## 研究問題與假設

### 主要研究問題

> 在保持輕量特徵提取的前提下，圖內／跨圖上下文互動與全局座標細化，哪一部分最能改善雙視圖配對，且計算成本是否值得？

### 可驗證假設

- **H1：**一層圖內 Self-Attention 能讓粗特徵使用更多場景上下文，降低重複紋理造成的錯配。
- **H2：**全局相關性加粗座標修正，會比只讀取固定候選位置更能降低初始座標誤差。
- **H3：**細尺度修正主要改善「已經在正確附近」的點，對一開始就搜尋錯區域的點幫助有限。
- **H4：**互動層數增加可能提高姿態準確率，但延遲會非線性增加；最佳方案不一定是最深的方案。

## 中性化的模型拆解

為了讓實驗先回答架構問題，本專案使用中性分支代號，不把任何現成系統名稱當作先驗答案，也不載入外部成品的預訓練權重。

### 共同特徵骨幹

兩張影像使用共享權重的輕量 CNN，輸出兩種尺度：

- **粗尺度特徵（約 1/8）**：提供大範圍搜尋與全局相關性。
- **細尺度特徵（約 1/4）**：提供局部座標修正。

這個骨幹是所有分支的共同控制條件；先不更換骨幹，避免把 backbone 差異誤認成匹配器差異。

### 軸 A：互動深度

| 代號 | 模組內容 | 要回答的問題 |
|---|---|---|
| `I0` | 描述子投影＋匹配／assignment head | 只有基本匹配頭能做到什麼程度？ |
| `I1` | 加 1 層圖內 Self-Attention | 少量上下文是否有幫助？ |
| `I2` | 1 層圖內互動＋1 層跨圖互動 | 跨圖訊息是否比圖內訊息更重要？ |
| `I4` | 重複 2 次圖內／跨圖互動 block | 深度增加是否持續有效？ |
| `I6` | 重複 6 次圖內／跨圖互動 block | 高準確率是否值得高延遲？ |

### 軸 B：定位階段

| 代號 | 模組內容 | 要回答的問題 |
|---|---|---|
| `L0` | 全局 correlation＋softmax 座標期望 | 只靠全局相關性可否得到合理初始位置？ |
| `L1` | `L0`＋粗尺度 CNN 座標修正 | 粗修正能否救回初始座標誤差？ |
| `L2` | `L1`＋細尺度 CNN 修正 | 細修正對像素級定位還有多少增益？ |

### Frozen／Unfrozen

- **Frozen**：固定共同特徵骨幹，只訓練後端互動／定位模組。
- **Unfrozen**：共同特徵骨幹與後端一起訓練。
- `L0-Frozen` 是必要的 forward-only 控制：它沒有可訓練的定位頭，因此不應被誤解成「訓練失敗」。

因此完整拆解矩陣是 16 個 cell：

```text
互動軸：I0、I1、I2、I4、I6       × Frozen／Unfrozen
定位軸：L0、L1、L2                × Frozen／Unfrozen
```

## 既有 pilot：只作動機與比較範例

以下數字來自先前相同研究脈絡的 78,042-step、464 對歷史開發集；它們不是本專案新訓練的成果，也不是未見場景保證。Frozen／Unfrozen 的數字只用來幫組員理解拆解邏輯。

### 互動深度軸

| 分支 | Frozen AUC@5 | Unfrozen AUC@5 | Frozen CPU s／pair | Unfrozen CPU s／pair |
|---|---:|---:|---:|---:|
| `I0` | 27.82% | 20.00% | 1.139 | 1.448 |
| `I1` | 32.55% | 37.42% | 1.469 | 1.678 |
| `I2` | 35.33% | 38.15% | 1.947 | 2.231 |
| `I4` | 36.63% | 42.72% | 2.795 | 3.041 |
| `I6` | 46.85% | 49.10% | 6.128 | 6.390 |

先前觀察：互動更深通常提高 AUC，但成本也上升；解凍也不保證每個分支都變好。這支持「做拆解」而不是直接選最深模型。

### 定位階段軸

| 分支 | Frozen AUC@5 | Unfrozen AUC@5 | Frozen CPU s／pair | Unfrozen CPU s／pair |
|---|---:|---:|---:|---:|
| `L0` | 5.59% | 7.92% | 1.507 | 1.525 |
| `L1` | 5.27% | 15.65% | 1.981 | 1.992 |
| `L2` | 7.39% | 16.56% | 2.127 | 2.135 |

先前觀察：在解凍條件下，粗修正帶來的增量（7.92%→15.65%）大於再加入細修正的增量（15.65%→16.56%）。這只是「粗尺度值得優先研究」的證據，不代表細修正沒有價值。

## 實驗資料與評估協定

### 資料

- 多視角建築／場景影像，附深度、相機與幾何資訊。
- 訓練以 image-pair exposure 計算；batch size 為 8 對影像。
- 既有紀錄的完整目標是每個可訓練分支 2,000,000 次 exposure；這是累計取樣次數，不是 2,000,000 組獨立影像對。
- 先前長訓最高提交到 89,730 steps；有完整 AUC 紀錄的 checkpoint 是 15,078 與 78,042 steps。

### 統一指標

1. **Pose AUC@5／@10／@20**：將配對交給固定姿態估計器，計算姿態誤差曲線下面積；不是單一通過率，也不是點配對正確率。
2. **correct@3／correct@5**：配對座標與幾何真值的像素誤差門檻。
3. **CPU 前端時間**：固定 1 CPU／0 GPU，報告秒／pair 的平均、p50、p95；不含姿態求解。
4. **單張特徵時間**：另記讀圖、前處理、特徵提取，不與兩圖 pipeline 混在同一欄。
5. **資源與可重現性**：參數量、峰值記憶體、輸入解析度、seed、checkpoint、資料 hash。

所有 cell 必須固定影像對、解析度、資料分割、姿態估計器、solver seeds 與計時邊界。AUC 只能在同一協定與同一資料集內比較。

## 四組最終消融

完成 16-cell 階梯後，才組合下列四組混合模型：

| 組別 | 上下文互動 | 座標修正 | 目的 |
|---|---|---|---|
| `baseline` | 無 | 無 | 建立共同起點 |
| `context_only` | 有 | 無 | 測搜尋／粗匹配是否改善 |
| `refinement_only` | 無 | 有 | 測座標定位是否改善 |
| `full_hybrid` | 有 | 有 | 測兩者是否互補 |

概念流程：

```text
Image A/B
  → shared lightweight CNN
  → coarse + fine feature maps
  → optional one-layer context block
  → global correlation
  → initial continuous coordinate
  → optional coarse refinement
  → optional fine refinement
  → final correspondence
  → fixed geometric pose evaluation
```

每次只改一個因素；若結果變差，記錄是搜尋、定位、監督、資料或成本問題，不用「模型不行」概括。

## 實驗流程

### Phase 0：固定基準

- 建立資料 manifest、scene split、pair hash 與 geometry sanity check。
- 固定輸入解析度、batch、optimizer、loss、seed、checkpoint 格式。
- 先用 8 對小樣本做 import／forward／backward 檢查，不把 smoke 分數當正式結果。

### Phase 1：16-cell 拆解

- 先跑 `I0→I1→I2→I4→I6` 與 `L0→L1→L2`。
- 每個 cell 保存 loss、AUC、correct@3、CPU timing、GPU memory、設定與 git commit。
- `L0-Frozen` 保留為固定控制，不把它列為可訓練成果。

### Phase 2：四組混合消融

- 固定同一資料與同一 evaluation protocol。
- 只比較「上下文」與「座標修正」是否各自有效、是否互補。
- 以準確率增益／延遲增量／記憶體成本一起選模型，不以單一 AUC 排名。

### Phase 3：失敗分析與報告

- 分類搜尋失敗、粗定位失敗、細定位失敗、姿態求解失敗。
- 對 rescued points、broken points、無候選點與大範圍錯配分開統計。
- 最後才討論是否加入額外監督或知識蒸餾；不能讓蒸餾掩蓋基礎架構的因果比較。

## 成功與停止條件

### 進入下一階段

- 同一協定下，至少一個模組相對 baseline 的 AUC@5 有可重現提升。
- 提升不是只出現在單一場景，且 solver seed 範圍沒有完全重疊到無法判讀。
- 速度／記憶體成本有明確記錄，組員能解釋提升來自搜尋還是定位。

### 停止或退回診斷

- 只有 loss 下降，AUC 與 correct@3 沒改善。
- 只改善已知候選子集，卻沒有改善完整資料。
- AUC 提升但 CPU／GPU 成本超過課程專題可展示範圍。
- 資料 hash、split、計時邊界或 checkpoint 不一致。

## 組員分工建議

| 角色 | 負責內容 | 必須交付 |
|---|---|---|
| 資料與評估 | manifest、split、幾何真值、AUC／correct 指標 | `data/`、evaluation script、protocol.md |
| 互動深度 | `I0/I1/I2/I4/I6` 與 frozen/unfrozen | model code、config、loss／timing |
| 定位階段 | `L0/L1/L2` 與座標誤差分析 | refiner code、rescue／failure tables |
| 系統與可重現性 | training entrypoint、checkpoint、環境與執行紀錄 | run manifest、requirements、reproduce.md |
| 分析與簡報 | 圖表、消融比較、限制、proposal／final presentation | figures、slides、member contributions |

每位組員都要在 commit、experiment record 或報告中留下可辨識的實際貢獻；不要只以「參加討論」作為分工紀錄。

## Repository 規劃

```text
ML_FP/
├── README.md                 # 本文件：問題、方法、協定與進度
├── src/                      # backbone、interaction、correlation、refinement
├── scripts/                  # train、evaluate、profile、failure analysis
├── configs/                  # 每個 cell 的固定設定
├── tests/                    # shape、gradient、determinism、metric tests
├── experiments/              # 每次 run 的 config、log、result、README
├── figures/                  # 報告用圖（不放大型原始資料）
└── requirements.txt          # 可重建環境
```

不要把 API key、私密資料、整包影像、虛擬環境或大型暫存檔 commit 到 repository。大型 checkpoint 只保存外部路徑、hash、step 與取得方式。

## 目前狀態

### 已有的研究基礎

- 已完成互動深度與粗細定位兩條拆解軸的 pilot。
- 已有 464 對開發集的 AUC 與 CPU 前端時間紀錄。
- 已知粗定位與候選座標是值得分開分析的失敗來源。
- 已規劃四組混合消融與共同 evaluation protocol。

### 尚未宣稱的成果

- 四組混合模型尚未完成正式訓練與比較。
- 尚未證明能達到任何外部系統的準確率或速度。
- 尚未完成未參與選型場景的泛化測試。
- pilot 數字不能取代 final report 的完整實驗。

## 第一週實作清單

1. 確認組員、資料使用權與可用計算資源。
2. 建立固定 manifest 與四組 baseline protocol。
3. 先完成 `L0-Frozen`、`L0-Unfrozen` 的 shape／gradient／小樣本 smoke。
4. 加入 `L1`，只比較粗座標修正的增益與成本。
5. 將每次結果寫成 `experiments/YYYYMMDD_<cell>/README.md`，包含方法、數據、限制與下一步。
6. 在 proposal video 中只報告動機、假設、拆解設計與評估計畫；不要把 pilot 當成混合模型成果。

## 可重現原則

每個結果至少要能回答：

- 使用哪個 commit、哪個 config、哪個資料 manifest？
- 實際更新幾步、看過多少 image-pair exposure？
- AUC 的資料分割、姿態 solver seeds、聚合方式是什麼？
- 時間從哪裡開始、在哪裡結束？是否含讀圖、特徵、匹配、姿態？
- checkpoint 是否可恢復？結果是否能由原始 log 重算？

若答案不完整，結果標為 partial／diagnostic，不標為正式模型結論。

## License／資料使用

本 repository 只保存程式、設定、摘要與可重現紀錄；影像、深度、相機參數與外部權重須依原始授權與課程規定取得。上傳前先確認資料不可公開時不會被 commit。

