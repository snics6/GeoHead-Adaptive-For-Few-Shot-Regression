# GeoHead Adaptation for Few-shot Regression — Design Doc

本ドキュメントは，`references/discussion.md` で整理された研究方針および
`references/Nejjar et al. 2023 - arXiv [cs.CV].pdf` (DARE-GRAM) を踏まえて，
toy 実験を実装する前の **設計仕様 / 実験プロトコル** を確定するものである．

---

## 1. 研究のポジショニング

### 1.1 中心問題

> **少数のターゲットラベルから，target 側の条件付き予測関数 $E_t[Y\mid Z]$ をどう復元するか**

を問う．最終目標は **shared regressor を成立させること** ではない．
むしろ conditional shift が一般に残ることを前提とし，

> **source-derived head を初期値として，few-shot target labels で
> target-optimal predictor へ素早く到達できる表現と適応則を学ぶこと**

を目標とする．

### 1.2 DARE-GRAM の役割の再定義

DARE-GRAM 原論文 (Nejjar et al., 2023) は

$$
P_s(Y\mid Z)\approx P_t(Y\mid Z)
$$

を暗に仮定して shared linear regressor を成立させる方向の手法．
本研究では損失式 (`Lsrc + αcos·Lcos + γscale·Lscale`) を **そのまま exact に再利用** するが，
意味付けを次のように変える:

> **shared regressor を作る regularizer ではなく，
> few-shot target support から head を補正しやすい
> representation geometry を学ぶ regularizer**

### 1.3 学習設定（重要・multi-source meta-learning + test-time few-shot adaptation）

本研究では以下の設定を採用する:

- **学習コーパス（3 つ，全て labeled）**: $D_1, D_2, D_3$．
  各 $D_i$ は自分の分布 $(\mu_i, \Sigma_i^{(x)}, \beta_i^\star)$ を持つ．
  各コーパスは独立に生成し，学習時はすべてのラベルにアクセス可．
- **テストコーパス（2 つ，held-out）**: $T_1, T_2$．
  学習中は**一切 touch しない**（x も y も使わない）．
- **Meta-learning の episode**: $\{D_1, D_2, D_3\}$ から **順序対 $(i, j)$** を $i \ne j$ でサンプル．
  - **inner の support** $S$ は $D_i$ から
  - **outer の query** $Q$ は $D_j$ から
  - すなわち **support と query は別コーパス**から取る（cross-corpus head adaptation）．
- **DARE-GRAM**: 同じ episode の $D_i$ と $D_j$ の表現間で計算する
  （"source=i, target=j" として原論文の式をそのまま適用）．
- **Test 時の few-shot adaptation**: テストコーパス $T_k$ から少数 support
  $\{(x, y)\}_{i=1}^k$ を抽出し，training と同じ inner rule で head を補正，残りで評価．

この設定は **multi-source domain generalization + few-shot adaptation** に相当する．

> 主張のロジック: meta-learning が学ぶのは
> "support corpus $D_i$ から few sample で補正した head を，
> 別コーパス $D_j$ の query でうまく動かす" 能力であり，
> これは本質的に **cross-domain head adaptation** の simulation．
> この能力が **未知の test corpus $T_1, T_2$ 上の few-shot adaptation
> に転移する** と期待する．
> 訓練中 test corpus は一度も見ていないので，
> "見たことのある target に過適合した" という反論を completely 回避できる．

---

## 2. 問題設定と記号

| 記号 | 意味 |
|---|---|
| $x \in \mathbb{R}^{d_x}$ | 入力 |
| $\phi_\theta : \mathbb{R}^{d_x}\to\mathbb{R}^p$ | 深層 encoder (toy では small MLP) |
| $z = \phi_\theta(x) \in \mathbb{R}^p$ | 表現 |
| $\beta \in \mathbb{R}^p$ | 線形 head |
| $\beta_0 \in \mathbb{R}^p$ | meta-initial head（学習対象） |
| $\hat y = \beta^\top z$ | 予測 |
| $D_1, D_2, D_3$ | training corpora (labeled, 学習で使用) |
| $T_1, T_2$ | test corpora (labeled, test-time の few-shot adaptation のみで使用) |
| $d$ | 一般的な domain index; §5 (DARE-GRAM) では $d \in \{s, t\}$ = episode の $(i, j)$ |
| $\Sigma_d = \mathbb{E}_d[ZZ^\top]$ | domain $d$ の二次モーメント行列 |
| $c_d = \mathbb{E}_d[ZY]$ | feature-label coupling |
| $\beta_d^\star = \Sigma_d^{-1} c_d$ | domain-optimal linear head |
| $S, Q$ | episode の support / query（train 時は training corpus 由来，test 時は test corpus 由来） |
| $\hat\Sigma_S$ | support / batch から推定した二次モーメント |

target risk（任意の domain $d$ 上で）:

$$
R_d(\beta) = \mathbb{E}_d[(Y - \beta^\top Z)^2]
$$

---

## 3. 理論的支柱（実装と評価指標の基礎）

### 3.1 risk decomposition

target 線形 conditional mean $m_t(Z) = Z^\top \beta_t^\star$ を仮定すると，

$$
R_t(\beta) = \mathbb{E}_t[\varepsilon_t^2] + (\beta - \beta_t^\star)^\top \Sigma_t (\beta - \beta_t^\star).
$$

特に source-optimal head を適用したときの excess risk:

$$
R_t(\beta_s^\star) - R_t(\beta_t^\star) = (\beta_s^\star - \beta_t^\star)^\top \Sigma_t (\beta_s^\star - \beta_t^\star).
$$

**意味**: head mismatch そのものではなく，target geometry $\Sigma_t$ で重み付けられた mismatch が悪化要因．

### 3.2 head mismatch の geometry / coupling 分解

$$
\beta_s^\star - \beta_t^\star = \underbrace{(\Sigma_s^{-1} - \Sigma_t^{-1}) c_s}_{A:\ \text{geometry shift}} + \underbrace{\Sigma_t^{-1}(c_s - c_t)}_{B:\ \text{coupling shift}}.
$$

これを risk に代入:

$$
R_t(\beta_s^\star) - R_t(\beta_t^\star) = A^\top \Sigma_t A + 2 A^\top \Sigma_t B + B^\top \Sigma_t B.
$$

**実装上の含意**:
- toy では $A, B$ を **独立に制御** できるよう設計する（節 6 参照）．
- 評価では `head correction size` ($\|\beta'-\beta_0\|_2$, $(\beta'-\beta_0)^\top \hat\Sigma_t (\beta'-\beta_0)$) を計測対象とする．

### 3.3 few-shot adaptation の解釈

few-shot adaptation を，

> support $S_t$ によって $\beta_0$ (source-derived) から $\beta_t^\star$ への
> 補正をどれだけ効率よく行えるか

という問題として定式化する．
"効率" は (i) 必要な support サイズ $k$，(ii) 補正の geometry-aware ノルム
$(\beta'-\beta_0)^\top \hat\Sigma_t (\beta'-\beta_0)$ で測る．

---

## 4. 提案手法 (GeoHead): bilevel / MAML 風

### 4.1 学習対象パラメータ

- encoder $\theta$
- meta-initial head $\beta_0$（training corpora から学ぶ shared initial head）

### 4.2 β_0 の初期化（warm-up）

bilevel 学習の収束を安定させるため，`β_0` は次の手順で初期化する:

1. **warm-up phase**: 3 training corpora の **pooled data** で，encoder $\theta$ と単一 head $\beta$ を
   普通の supervised regression で学習（pooled MSE のみ，適応なし）．
2. warm-up 終了時の $\beta$ を `β_0` の初期値，$\theta$ の状態を encoder の初期値とする．
3. その後，§4.3〜§4.5 の bilevel 学習を行う．

> 注: pooled training は各 corpus の β_i^\star が異なるため完全には fit できないが，
> "全 corpus で平均的に妥当な head" を得る粗い初期化として機能する．

### 4.3 inner loop (per episode, cross-corpus) — head only (ANIL)

各 episode で順序対 $(i, j)$ を $\{D_1, D_2, D_3\}$ から $i \ne j$ でサンプル．

- $S \sim D_i$: support（inner で使う，corpus $i$ から）
- $Q \sim D_j$: query（outer で使う，**別コーパス** $j$ から）
- $B_i \sim D_i$, $B_j \sim D_j$: DARE-GRAM 用のフル batch（ラベル必須ではない，$x$ のみでも可）

inner では **head $\beta$ のみ update**（encoder $\theta$ は固定）．
これは ANIL (Almost-No-Inner-Loop) 風で，§3 の risk decomposition
（head に対する quadratic）と最も整合的．

inner loss:

$$
L_{\text{inner}}(\beta;\theta, S) = L_{\text{sup-pred}} + \lambda_h L_{\text{head-reg}}
$$

$$
L_{\text{sup-pred}} = \frac{1}{|S|}\sum_{(x,y)\in S}(y - \beta^\top \phi_\theta(x))^2
$$

$$
L_{\text{head-reg}} = (\beta - \beta_0)^\top (\hat\Sigma_{S} + \varepsilon I)(\beta - \beta_0)
$$

ここで $\hat\Sigma_{S} = \frac{1}{|S|}\sum_{x\in S} z z^\top$ は support 上の二次モーメント．

inner update: **複数ステップの勾配降下**（MAML 標準）

$$
\beta^{(k+1)} = \beta^{(k)} - \eta_\text{in} \nabla_\beta L_{\text{inner}}(\beta^{(k)}; \theta, S), \quad k = 0,\ldots,K-1
$$

初期値 $\beta^{(0)} = \beta_0$．

> 採用方針: head-only ANIL なので外部ライブラリは不要で，**PyTorch の
> `torch.autograd.grad(..., create_graph=True)` で inner update を
> differentiable に展開**し，二階微分を outer に流す（first-order 近似は
> ablation で比較可）．ANIL なので二階微分のコストは head に対してのみ発生し，
> 軽い．**実装上は train と test で全く同じ inner rule**
> (`geohead.adaptation.test_time.inner_rule_adapt`) を再利用する（`create_graph=True`
> が train，`create_graph=False` が test）ことで，§8.3 の train/test 一貫性要件を
> 満たす．

### 4.4 outer loop

outer loss は **cross-corpus query MSE** と **DARE-GRAM regularizer** を組み合わせる．
DARE-GRAM は **outer 側のみ** に置く（inner には置かない）:

$$
L_{\text{outer}}^{(i,j)} = L_{\text{qry}}(\beta'; Q, \theta) + \lambda_D \cdot L_{\text{DARE}}(B_i, B_j; \theta)
$$

$$
L_{\text{qry}} = \frac{1}{|Q|}\sum_{(x,y)\in Q}(y - (\beta')^\top \phi_\theta(x))^2
$$

$$
L_{\text{DARE}}(B_i, B_j; \theta) = L_{\text{src}}(B_i) + \alpha_{\cos} L_{\cos}(Z(B_i), Z(B_j)) + \gamma_{\text{scale}} L_{\text{scale}}(Z(B_i), Z(B_j))
$$

ここで:
- $\beta'$ は inner update 後の head（$\theta$ と $\beta_0$ に微分可能依存）．
- $L_{\text{src}}(B_i)$ は corpus $i$ での MSE（現 episode における "source"）．
  最終的には **全 corpus の MSE を平均** して supervised signal としても良い（sweep 対象; §14 留保事項）．
- $L_{\cos}, L_{\text{scale}}$ は corpus $i$ と corpus $j$ の表現間の inverse-Gram alignment．

outer の update 対象: $(\theta, \beta_0)$．

> **実装ノート**: `src/geohead/losses/dare_gram.py` の `dare_gram_regularizer` は
> **regularizer 成分のみ** $\alpha_{\cos} L_{\cos} + \gamma_{\text{scale}} L_{\text{scale}}$ を返し，
> $L_{\text{src}}$ は caller 側で別途計算する（GeoHead outer では $L_{\text{src}}(B_i)$,
> §8.1 baseline では pooled $L_{\text{src}}(B_i \cup B_j)$ と，使い分けが必要なため）．

### 4.5 学習スキーマ全体

```
warm-up phase:
    pool data from D_1, D_2, D_3
    minimize MSE w.r.t. (θ, β) on pooled data
    set β_0 := β,  θ := θ (carry over)

bilevel phase:
for outer_step in 1..N_outer:
    -- episode setup --
    sample ordered pair (i, j) with i != j from {1, 2, 3}
    sample support  S  of size |S|  from D_i
    sample query    Q  of size |Q|  from D_j
    sample batch    B_i (disjoint from S) from D_i   # for DARE-GRAM
    sample batch    B_j (disjoint from Q) from D_j   # for DARE-GRAM

    -- inner (head only, K steps GD) --
    β' = inner_update(β_0, θ, S, K)

    -- outer --
    L_qry  = MSE(Q, β', θ)
    L_DARE = DARE_GRAM(Z(B_i), Z(B_j), Y(B_i))
    L_total = L_qry + λ_D · L_DARE
    update (θ, β_0) via Adam
```

> 注意: test corpora $T_1, T_2$ は **一切登場しない**（x も y も）．
> test corpora のラベルは test 時の few-shot adaptation 評価でのみ使う（§8）．

### 4.6 episode 組み立ての補足

- 順序対 $(i, j)$ は各 step でランダムサンプル．3 corpora で $3 \times 2 = 6$ 通り．
- 1 epoch で 6 通りを均等に回すか，確率均等サンプルかは実装詳細（後者を採用）．
- batch 内の $S$ と $B_i$ は **disjoint** に取る（情報リークを避ける）．

---

## 5. DARE-GRAM の正確な定式化（exact reproduction）

論文 §3.3–§3.5, Eq.(6)–(12) に厳密に従う．
以下で "source / target" は論文の記法に合わせた abstract な 2 ドメイン名であり，
本研究の multi-corpus 設定では **episode 毎にサンプルした順序対 $(i, j)$** のうち
$i$ を "source"，$j$ を "target" として損失を計算する（§4.4 参照）．

### 5.1 Gram matrix と pseudo-inverse

batch 内 feature $Z_d \in \mathbb{R}^{b\times p}$ について，

$$
G_d := Z_d^\top Z_d \in \mathbb{R}^{p\times p}.
$$

SVD: $Z_d = U_d D_d V_d^\top$ より，

$$
G_d = V_d \Lambda_d V_d^\top, \quad \lambda_{d,k} = D_{d,kk}^2.
$$

固有値を降順に並べ，累積比による主成分本数 $k_d$ を選択:

$$
k_d = \min\left\{ k :\ \frac{\sum_{i=1}^k \lambda_{d,i}}{\sum_{i=1}^p \lambda_{d,i}} > T \right\}.
$$

**両ドメインに共通の** 主成分数として `k = max(k_s, k_t)` または論文式 (8) のように
両条件を同時に満たす最小 `k` を採用する（論文準拠で後者）．

pseudo-inverse:

$$
G_d^+ = V_d\, \mathrm{diag}\!\left(\tfrac{1}{\lambda_{d,1}},\ldots,\tfrac{1}{\lambda_{d,k}}, 0,\ldots,0\right) V_d^\top.
$$

### 5.2 angle alignment (cosine)

各列 $G_{s,i}^+$, $G_{t,i}^+$ について，

$$
\cos(\theta_i^{S\leftrightarrow T}) = \frac{G_{s,i}^+ \cdot G_{t,i}^+}{\|G_{s,i}^+\|\,\|G_{t,i}^+\|}, \quad i = 1,\ldots,p.
$$

$$
M = [\cos\theta_1, \ldots, \cos\theta_p] \in \mathbb{R}^p, \quad
L_{\cos} = \|\mathbf{1} - M\|_1.
$$

> **実装ノート**: scale 非依存性を厳密に保つため，実装 (`dare_gram_regularizer`) では
> 各列を `F.normalize(·, p=2, dim=0, eps=ε)` で単位化してから内積を取る．
> これは `dot / (‖a‖·‖b‖ + ε)` 方式に存在する $O(\varepsilon / \|a\|\|b\|)$ の
> スケール依存バイアスを回避するためである．

### 5.3 scale alignment (eigenvalues)

選ばれた `k` 個の主固有値について:

$$
L_{\text{scale}} = \left\| \lambda_{s,1:k} - \lambda_{t,1:k} \right\|_2.
$$

### 5.4 source supervised loss

$$
L_{\text{src}} = \frac{1}{N_s}\sum_{i=1}^{N_s}(y_i - \tilde y_i)^2.
$$

### 5.5 total loss

$$
L_{\text{total}} = L_{\text{src}} + \alpha_{\cos} L_{\cos} + \gamma_{\text{scale}} L_{\text{scale}}.
$$

本研究の実装では $L_{\text{total}}$ そのものを単一関数で提供せず，
`dare_gram_regularizer` は regularizer 成分 $\alpha_{\cos} L_{\cos} + \gamma_{\text{scale}} L_{\text{scale}}$
のみを返し，$L_{\text{src}}$ は caller が組み立てる（§4.4 実装ノート参照）．

### 5.6 数値安定化

- SVD は `torch.linalg.svd` を使用．
- 微分の安定性のため，固有値に floor `max(λ, ε)` を入れる（ε ≈ 1e-6）．
- batch サイズ `b` < `p` のケースを想定（pseudo-inverse の意義）．
- 累積比率計算（$k_d$ 選択）は `torch.no_grad()` で行う（discrete argmax は微分不要）．

---

## 6. Toy データ生成プロトコル

### 6.1 共通の真の表現

固定された MLP $\phi^\star : \mathbb{R}^{d_x}\to \mathbb{R}^{p^\star}$ を生成し，
全 domain で共有する（参照点）．

```python
phi_star = MLP(d_x -> 64 -> 32 -> p_star),  random init, frozen
```

### 6.2 domain ごとの分布

domain $d$ について:

$$
x \sim \mathcal{N}(\mu_d, \Sigma_d^{(x)}), \quad
y = (\beta_d^\star)^\top \phi^\star(x) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2).
$$

- $(\mu_d, \Sigma_d^{(x)})$ を変える → covariate shift（→ representation 上では geometry shift）
- $\beta_d^\star$ を変える → conditional shift（→ head shift）

### 6.3 用意する domain と shift 強度

#### Training corpora（3 つ，全て labeled）

基準ベクトル $\beta_{\text{base}} \sim \mathcal{N}(0, I_{p^\star})$ を一度だけ生成・固定．
また 3 つの独立な方向ベクトル $\delta_1, \delta_2, \delta_3$（正規化，基準と非相関）も固定．

| Corpus | $\mu$ | $\Sigma^{(x)}$ | $\beta^\star$ |
|---|---|---|---|
| $D_1$ | $\mathbf{0}$ | $I$ | $\beta_{\text{base}}$ |
| $D_2$ | $\mu_1$ (shifted) | $R_1 \mathrm{diag}(s_1) R_1^\top$ | $\beta_{\text{base}} + 0.4\,\delta_1$ |
| $D_3$ | $\mu_2$ (shifted) | $R_2 \mathrm{diag}(s_2) R_2^\top$ | $\beta_{\text{base}} + 0.4\,\delta_2$ |

ここで:
- $\|\mu_i\| \approx 0.5$, ランダム方向（seed 固定）
- $R_i$: ランダム直交行列（seed 固定）
- $s_i$: 対数正規分布で生成，`exp(N(0, 0.3^2))` 程度
- 方向 $\delta_i$ はすべて互いに直交 ($\delta_1 \perp \delta_2 \perp \delta_3$)

→ training corpora は **moderate shift** で，3 コーパス間の差異は encoder が学ぶ domain invariance + head adaptability の "訓練信号" になる．

#### Test corpora（2 つ，training 中は一切 touch しない）

| Corpus | 目的 | 構成 |
|---|---|---|
| $T_1$: `interp` | 訓練 corpora の shift 範囲内 | $\mu, \Sigma$: $D_2, D_3$ の線形補間 / $\beta^\star = \beta_{\text{base}} + 0.4 \cdot (0.5 \delta_1 + 0.5 \delta_2)$ |
| $T_2$: `extrap` | 訓練 corpora の shift 範囲外 | 新規方向 $\delta_4$（$\delta_1, \delta_2, \delta_3$ とは別）で大きな head shift: $\beta^\star = \beta_{\text{base}} + 0.8 \delta_4$ / 加えて geometry も大きく異なる |

- $T_1$ は "訓練コーパスから interpolate できるが，見たことはない" 条件 → **interpolation generalization**
- $T_2$ は "訓練コーパスから extrapolate する必要がある" 条件 → **extrapolation robustness**

期待する結果:
- 提案手法は $T_1$ で明確に baseline を上回る（meta-learning の直接的な恩恵）
- $T_2$ では提案手法の優位が減るかもしれない（extrapolation の限界）
  → これは負の結果として誠実に報告する

#### Test-time 評価プロトコル

各 test corpus $T_k$ について:
- support pool: 200 サンプル（ラベル付き）
- query pool: 1000 サンプル（ラベル付き，held-out）
- support size を $k \in \{1, 3, 5, 10, 20\}$ で振り，各 $k$ で support pool から
  ランダムサンプリングしたうえで adaptation → query で MSE/MAE 計測を複数 seed 繰り返し．

### 6.4 サイズとデータ構造

- `d_x = 16`, `p_star = 32`
- 各 training corpus $D_i$: **labeled 5000 サンプル**（内訳: warm-up 用 pooled, episode 用 pool はその場で split）
- 各 test corpus $T_k$: support pool 200 + query pool 1000（合計 1200，ラベル付き）
- support size: $k \in \{1, 3, 5, 10, 20\}$（test 時に sub-sampling，各 $k$ で 20 seed）

**データ API（実装）**: `build_toy_dataset(...)` は `ToyDataset` を返し，

```python
ds.train[name]                 # (X, Y) for name in {"D1", "D2", "D3"}
ds.test[name]["support"]       # (X, Y) shape=(200,  d_x), (200,)
ds.test[name]["query"]         # (X, Y) shape=(1000, d_x), (1000,)
```

test corpora は生成時点で support/query に分割されており，test-time adaptation は
`ds.test[name]["support"][:k]` を few-shot support，`ds.test[name]["query"]` を
評価対象として使う．

---

## 7. モデル設計

```text
Encoder φ_θ:  MLP(d_x=16 -> 64 -> 64 -> p=32),  ReLU + LayerNorm
Head β:       Linear(p=32 -> 1),  no bias  (理論との整合のため)
Init head β_0: Linear(p=32 -> 1),  meta-learned
```

`p=32` を `p_star=32` と一致させるが，encoder は学習対象．
出力 dim を 1 に絞ることで理論式 ($\beta \in \mathbb{R}^p$) と直接対応．

---

## 8. 比較対象（3本比較プロトコル）

すべて **同じ training データ条件**（3 training corpora $D_1, D_2, D_3$ のみ使用，
test corpora $T_1, T_2$ は学習中に触らない），**同じ encoder アーキテクチャ**，
**同じ評価プロトコル** で比較する．

### 8.1 Baseline 1: `DARE+ridge`

- **Train**: meta-learning なし．単純な multi-source + DARE-GRAM training．
  - 各 step で順序対 $(i, j)$ を $\{D_1, D_2, D_3\}$ から $i \ne j$ でサンプル
  - **pooled supervised MSE**: $L_{\text{src}} = \frac{1}{|B_i|+|B_j|}\sum \text{MSE}$ （両 corpus のラベルを使う）
  - **DARE-GRAM**: $L_{\cos}(Z(B_i), Z(B_j)) + L_{\text{scale}}(\ldots)$
  - 最終 loss: $L_{\text{src}} + \alpha_{\cos} L_{\cos} + \gamma_{\text{scale}} L_{\text{scale}}$（$\lambda_D$ は掛けず，$\alpha_{\cos}, \gamma_{\text{scale}}$ 自身で重み調整）
- **学習後の head**: training 終了時点の $\beta$ を `β_0` として保存（encoder $\theta$ も保存）．
- **Test-time adaptation** on test corpus $T_k$: support $S_t \subset T_k$ で closed-form ridge 解を求める:

$$
\hat\beta = \arg\min_\beta \sum_{(x,y)\in S_t}(y-\beta^\top \phi_\theta(x))^2 + \lambda \|\beta - \beta_0\|^2
$$

> **実装**: `src/geohead/training/baseline.py::baseline_train` が学習ループ，
> `src/geohead/adaptation/test_time.py::ridge_adapt` が test-time adaptation．

### 8.2 Baseline 2: `DARE+geo`

- **Train**: Baseline 1 と完全に同じ．
- **Test-time adaptation のみ違う**: geometry-aware head reg で closed-form 解:

$$
\hat\beta = \arg\min_\beta \sum_{(x,y)\in S_t}(y-\beta^\top \phi_\theta(x))^2 + \lambda(\beta - \beta_0)^\top(\hat\Sigma_t + \varepsilon I)(\beta - \beta_0)
$$

$\hat\Sigma_t$ は test corpus $T_k$ の support から推定（§8.3 inner-rule の $\hat\Sigma$ と整合）．

> **実装**: `src/geohead/adaptation/test_time.py::geo_adapt`（Baseline 1 と同じ学習済み
> encoder/head を共有し，adaptation のみ差し替え）．

### 8.3 Proposed: `GeoHead (full)`

- **Train**: §4 の bilevel/MAML 構成
  - warm-up phase で pooled $D_1,D_2,D_3$ から `β_0`, `θ` を初期化
    (`src/geohead/training/warmup.py::warmup_train`)
  - bilevel phase: episode $(i,j)$，inner on $D_i$，outer query on $D_j$，
    DARE-GRAM between $D_i,D_j$
  - inner: head only ANIL, K-step GD with geometry-aware head reg
  - outer: cross-corpus query MSE + DARE-GRAM
- **Test-time adaptation** on $T_k$: **training の inner rule をそのまま適用**
  （`K` step の gradient descent，geometry-aware head reg，$\hat\Sigma$ は test support から推定）．

> **実装**: warm-up・bilevel trainer はいずれも実装済み
> (`src/geohead/training/warmup.py::warmup_train`,
> `src/geohead/training/geohead.py::geohead_train`)．bilevel trainer は
> `inner_rule_adapt(create_graph=True)` を呼び出して inner loop を
> differentiable に展開し，outer 側で Adam 1 step で $(\theta, \beta_0)$ を
> 更新する．Test-time 側は同じ関数を `create_graph=False` で呼ぶだけなので，
> train/test の inner rule が厳密一致することはテスト
> (`tests/training/test_geohead.py::test_geohead_training_inner_rule_matches_test_time_inner_rule`)
> で担保されている．

### 8.4 追加 ablation（時間が許せば）

| Variant | warm-up | bilevel | inner head reg | outer DARE | 目的 |
|---|---|---|---|---|---|
| `Baseline-warmup-only` | あり | なし | — | — | β_0 warm-up 単体の効果 |
| `GeoHead w/o DARE` | あり | あり | あり | なし | DARE-GRAM の必要性 |
| `GeoHead w/o headreg` | あり | あり | naive ridge | あり | geometry-aware reg の必要性 |
| `GeoHead first-order` | あり | あり | あり | あり | 二階微分の必要性 |
| `GeoHead full-MAML` | あり | あり (encoder も inner で更新) | あり | あり | ANIL vs MAML |
| `GeoHead single-source` | あり (D1 のみ) | あり (D1 内 split) | あり | あり | multi-source の必要性 |

---

## 9. 評価指標

### 9.1 主指標
- **target query MSE** (理論との接続が直接)
- **target query MAE** (DARE-GRAM 論文との比較しやすさ)

### 9.2 補助指標
- adaptation 前後の query error
- **head correction size**
  - $\|\hat\beta - \beta_0\|_2$
  - $(\hat\beta - \beta_0)^\top \hat\Sigma_t (\hat\beta - \beta_0)$
- subspace alignment 可視化:
  - source/target Gram の主固有値スペクトル
  - inverse Gram の cosine similarity 行列
  - principal angles のヒストグラム

### 9.3 評価軸

各 (method, test corpus $T_k$, support size $k$) について:
- 20 seed （test-time に support を re-sample）の mean ± 95%CI
- 表: corpus × $k$ × method の query MSE
- 図: $k$ vs query MSE の sample efficiency curve（methods を色分け）
- 図: head correction size と query MSE の相関散布図（理論式の実証）

> **実装**: §9.1 / §9.2 の主指標と補助指標は
> `src/geohead/evaluation/metrics.py` に，§9.3 の評価行列は
> `src/geohead/evaluation/runner.py::evaluate_model` に，§9.3 の 2 つの
> 図は `src/geohead/evaluation/visualize.py` に実装．subspace alignment
> 可視化は M3 に回す（本体の sample-efficiency curve 後に追加）．
>
> **Fair-comparison invariant**: 同じ $(T_k, k, \text{seed})$ 三つ組の中で，
> `none / ridge / geo / inner` の 4 method はすべて同一の support
> sub-sample を見る（seed 固定の `torch.randperm` で決定論的に抽出）．
> これにより method 差が support 抽選ノイズから分離される．
> テスト
> (`tests/evaluation/test_runner.py::test_evaluate_model_all_methods_see_same_support_sample`)
> で担保．
>
> **`sigma_hat` の出どころ**: §9.2 の $\hat\Sigma_t$ は「test 時に利用可能な
> target feature」のみから推定する必要があるので，**test support の
> $k$ サンプルから** `second_moment(z_sup)` を計算し，geo head-reg / inner
> rule / 補助指標 `delta_geo` のすべてで共用する．

---

## 10. ハイパーパラメータ初期値

| カテゴリ | パラメータ | 初期値 |
|---|---|---|
| encoder | hidden dim, output dim | 64, 32 |
| データ | `d_x`, `p_star` | 16, 32 |
| データ | corpus size (each train) | 5000 |
| データ | test support / query pool | 200 / 1000 |
| 学習 | optimizer | Adam |
| 学習 | warm-up epochs | 20 |
| 学習 | warm-up lr | 1e-3 |
| 学習 | warm-up batch_size (pooled) | 256 |
| 学習 | bilevel outer steps | 10000 |
| 学習 | baseline outer steps | 10000 |
| 学習 | outer lr (θ, β_0) | 1e-3 |
| 学習 | episode batch (S, Q, $B_i$, $B_j$) | 32, 64, 64, 64 |
| 学習 | baseline batch ($B_i$, $B_j$) | 64, 64 |
| inner | inner lr `η_in` | 0.1 |
| inner | inner steps `K` | 5 |
| inner | head reg `λ_h` | 0.1 |
| DARE | `α_cos` | 0.01 (論文 Fig.8 中央値付近) |
| DARE | `γ_scale` | 1e-4 (同上) |
| DARE | threshold `T` | 0.99 |
| DARE | outer 重み `λ_D` | 1.0 |
| 数値 | ε (Σ floor) | 1e-6 |
| test-time | adaptation steps | 5 (training inner と同じ) |
| test-time | support sizes | 1, 3, 5, 10, 20 |
| test-time | seeds per (method, corpus, k) | 20 |

→ 主要 sweep: `λ_h`, `λ_D`, `K`, `α_cos`, `γ_scale`．

---

## 11. リポジトリ構成（実装済み + 予定）

実装の進捗に合わせ，§11 の構成案は実コードベースの形に寄せた．`[x]` は実装済み，
`[ ]` は今後のマイルストーン（§12）で実装予定．

```
GeoHead-Adaptation-for-Few-shot-Regression/
├── README.md                            [x]
├── pyproject.toml                       [x]  (uv + pyenv-virtualenv)
├── uv.lock                              [x]
├── .python-version                      [x]  (pyenv local: geohead / 3.11.9)
├── docs/
│   └── design.md                        [x]  ← 本書
├── references/                          [x]  (discussion.md + DARE-GRAM PDF)
├── src/geohead/
│   ├── __init__.py                      [x]
│   ├── data/
│   │   ├── __init__.py                  [x]
│   │   ├── toy.py                       [x]  §6 のデータ生成
│   │   └── episode.py                   [x]  episode / DARE pair sampler (§4.4, §8.1)
│   ├── models/
│   │   ├── __init__.py                  [x]
│   │   ├── encoder.py                   [x]  MLPEncoder (Linear→ReLU→LayerNorm)
│   │   └── head.py                      [x]  LinearHead (no bias, .beta アクセス)
│   ├── losses/
│   │   ├── __init__.py                  [x]
│   │   ├── dare_gram.py                 [x]  §5 exact 実装 (regularizer のみ)
│   │   └── head_reg.py                  [x]  geometry-aware head regularizer
│   ├── adaptation/
│   │   ├── __init__.py                  [x]
│   │   └── test_time.py                 [x]  ridge_adapt / geo_adapt / inner_rule_adapt
│   ├── training/
│   │   ├── __init__.py                  [x]
│   │   ├── warmup.py                    [x]  §4.2 pooled supervised MSE
│   │   ├── baseline.py                  [x]  §8.1 DARE+ridge 学習ループ
│   │   └── geohead.py                   [x]  §4 bilevel meta-trainer (ANIL + DARE)
│   ├── evaluation/                      [x]  §9 eval suite
│   │   ├── __init__.py                  [x]
│   │   ├── metrics.py                   [x]  query_mse / mae, head_correction_l2 / geo
│   │   ├── runner.py                    [x]  evaluate_model (4-method × T_k × k × seed)
│   │   └── visualize.py                 [x]  sample-efficiency curve, delta-vs-MSE scatter
│   └── utils/                           [ ]
│       ├── seed.py                      [ ]
│       └── config.py                    [ ]
├── experiments/                         [ ]
│   ├── configs/
│   │   ├── baseline_dare_ridge.yaml     [ ]
│   │   ├── baseline_dare_geo.yaml       [ ]
│   │   └── geohead_full.yaml            [ ]
│   ├── scripts/
│   │   ├── run_baseline.py              [ ]
│   │   ├── run_geohead.py               [ ]
│   │   └── run_all.sh                   [ ]
│   └── results/                         [ ]  (gitignore)
└── tests/
    ├── __init__.py                      [x]
    ├── data/                            [x]  test_toy.py, test_episode.py
    ├── models/                          [x]  test_encoder.py, test_head.py
    ├── losses/                          [x]  test_dare_gram.py, test_head_reg.py
    ├── adaptation/                      [x]  test_test_time.py
    ├── training/                        [x]  test_warmup.py, test_baseline.py, test_geohead.py
    └── evaluation/                      [x]  test_metrics.py, test_runner.py, test_visualize.py
```

**構成の差分（原案→実装）**:

- `adaptation/ridge.py`, `inner_loop.py` → **`adaptation/test_time.py`** に統合．
  GD inner rule と closed-form ridge/geo は同じ $\beta$-空間で動く API (`z, y, β_0, ...` を受け取る)
  なので 1 ファイルにまとめた．`create_graph=True` フラグで train (bilevel) / test
  の両方から同じ `inner_rule_adapt` を再利用できる．
- `training/warmup.py` を追加（§4.2 を独立モジュールに）．
- `training/geohead.py` は当初 `higher` 依存を想定していたが，**head-only ANIL
  では外部ライブラリなしで同じ差分可能展開が書ける**ため，
  `torch.autograd.grad(..., create_graph=True)` のみで実装．train/test で
  inner rule が厳密一致する利点（§8.3）も得られる．
- `tests/` は機能別サブディレクトリに整理（`tests/<module>/test_*.py`）．

---

## 12. 実装マイルストーン

### M1: 基盤
- [x] toy data generator (`src/geohead/data/toy.py`)
- [x] DARE-GRAM loss の **単体テスト**（SVD, cos, scale, autograd, scale invariance）
- [x] encoder + head (`src/geohead/models/`)

### M2: ベースライン整備（サブマイルストーン）
- [x] **M2.1** geometry-aware head regularizer + 2nd-moment helper (`src/geohead/losses/head_reg.py`)
- [x] **M2.2** episode sampler（`sample_random_pair`, `sample_episode`, `sample_dare_pair`）
- [x] **M2.3** warm-up trainer (`src/geohead/training/warmup.py`)
- [x] **M2.4** test-time adaptation 三点セット (`ridge_adapt`, `geo_adapt`, `inner_rule_adapt`)
- [x] **M2.5** baseline 1 (`DARE+ridge`) training loop (`src/geohead/training/baseline.py`)
- [x] **M2.6** GeoHead bilevel meta-trainer (`src/geohead/training/geohead.py`)（head-only ANIL を `inner_rule_adapt(create_graph=True)` で unroll．`higher` には依存せず，train/test で同じ inner rule を共有）
- [x] **M2.7** evaluation runner (`src/geohead/evaluation/`)
  - `metrics.py` (query MSE/MAE, head correction L2 / geo)
  - `runner.py` (`evaluate_model`: 4-method × 2 corpora × 5 k-shot × 20 seed，fair-comparison invariant つき)
  - `visualize.py` (sample efficiency curve，head correction vs MSE scatter)
  - 4 method は `{none, ridge, geo, inner}`（`none` は β_0 のまま，§9.3 に加えて sanity 用）

### M3: 提案手法の sanity check
- [ ] proposed が toy で Baseline 1 / 2 を上回るか確認（最初の end-to-end 実験）

### M4: 実験本番
- [ ] 2 test corpora × 5 support sizes × 20 seed × 3 methods
- [ ] sample efficiency curve
- [ ] head correction size の集計

### M5: 分析と ablation
- [ ] subspace 可視化（Gram spectra, cos similarity, principal angles）
- [ ] ablation（`w/o DARE`, `w/o headreg`, `first-order`, `full-MAML`, `single-source`）

---

## 13. 主張ステートメント（避ける／使う）

### 避ける
- "source / target で shared regressor を成立させる"
- "conditional shift を消す"
- "feature alignment が究極の目標"

### 使う
- "source-derived head を target few-shot labels で迅速に補正可能な representation を学ぶ"
- "inverse-Gram geometry を整えることで few-shot head adaptation の sample efficiency を改善する"
- "representation geometry と head correction cost を joint に考慮する"

---

## 14. 留保事項 / 後で詰める点

- **`λ_D` の outer scale**: DARE-GRAM 原論文では single-stage training なので `α_cos` の値の意味が違う可能性．outer に置く際の重み調整を sweep で確認．
- **cross-corpus episode が test corpus に転移する根拠**: 本研究の核となる仮定．
  training corpora 間で学んだ "別コーパスへの head adaptation" 能力が，
  未知 test corpus $T_1, T_2$ でも有効に働くと期待．
  特に $T_2$（extrapolation）では効果が落ちる可能性があり，toy で転移の限界を計測する．
- **corpus の数**: 3 training corpora が "meta-task distribution" として十分な多様性を持つか．
  少なすぎると episode のバリエーションが限られる（$3 \times 2 = 6$ 通り）．
  結果次第で将来 4〜5 corpora に拡張．
- **`L_src` の対象 batch**: outer の supervised term を $B_i$ のみにするか $B_i \cup B_j$ にするか．
  前者は episode の方向性を明確にし，後者は全 corpus のラベル情報を活かす．デフォルトは前者．sweep 対象．
- **warm-up phase の epoch 数**: 短すぎると β_0 が bad init，長すぎると bilevel が局所解に陥る．試行錯誤．
- **inner の $\hat\Sigma$ 推定**: support のみ ($k=1$ だと退化)，support+unlabeled batch，
  または $\hat\Sigma_i$（全 corpus の pool）の 3 候補．test 時は support から推定するため，
  train/test で整合させる．
- **Baseline 1 の $\lambda_D$**: §8.1 では敢えて総損失を $L_{\text{src}} + \alpha L_{\cos} + \gamma L_{\text{scale}}$
  と定義し，$\lambda_D$ を掛けない（$\alpha_{\cos}, \gamma_{\text{scale}}$ 自身で重み調整）．
  GeoHead outer の $\lambda_D$ とは別概念であることに注意．baseline の実装は
  この方針に沿う（`BaselineConfig` に `lambda_D` を持たない）．
- **Test corpus の support/query pre-split**: `build_toy_dataset` は $T_k$ の生成時に
  support pool 200 と query pool 1000 を別シードで独立生成している（§6.4 データ API）．
  sub-sampling 時に「support pool から $k$ 個」を取れば，query pool と完全に i.i.d.
  だが独立なサンプルになる．これは test-time の sample efficiency 曲線を
  "同じ query で support の情報量だけ変化させた" 実験として解釈可能にする．
