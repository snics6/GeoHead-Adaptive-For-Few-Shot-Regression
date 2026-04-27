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

inner update: **複数ステップの（任意で前処理付き）勾配降下**（MAML 標準）

$$
\beta^{(k+1)} = \beta^{(k)} - \eta_\text{in} \, P \, \nabla_\beta L_{\text{inner}}(\beta^{(k)}; \theta, S), \quad k = 0,\ldots,K-1
$$

初期値 $\beta^{(0)} = \beta_0$．前処理行列 $P$ は次の二択:

- **vanilla GD**（`preconditioned=False`，backward-compat default）: $P = I$．
- **damped natural gradient**（`preconditioned=True`，M3 v5 以降の推奨）: $P = (\hat\Sigma_S + \varepsilon I)^{-1}$．すなわち，**§8.2 の `geo_adapt` と同じ計量**で勾配を前処理する．

**前処理の動機**: vanilla 勾配は
$\nabla L_{\text{inner}} = \tfrac{2}{|S|}Z^\top(Z\beta - y) + 2\lambda_h \hat\Sigma_S (\beta - \beta_0)$
で，第一項が $\|Z\|^2$ でスケールする．DARE-GRAM を経た encoder は $L_{\cos}, L_{\text{scale}}$ によって特徴量ノルムが増減しやすく，固定 $\eta_\text{in}$ だと発散の危険がある（M3 v4 の failure mode）．$(\hat\Sigma_S + \varepsilon I)^{-1}$ を掛けると $Z \to cZ$ に対して $\hat\Sigma_S \to c^2 \hat\Sigma_S$，$\nabla L \to c^2 \nabla L$，$P \to c^{-2} P$ で相殺し，**特徴量スケール不変**になる．極限 $\varepsilon \to 0, K=1, \eta_\text{in} = 1/2, \lambda_h = 0$ では 1 step で OLS 解に到達し，$\eta_\text{in} < 1/2$ なら ridge-like な安定軌道で $\beta_0$ から移動する．

> 採用方針: head-only ANIL なので外部ライブラリは不要で，**PyTorch の
> `torch.autograd.grad(..., create_graph=True)` で inner update を
> differentiable に展開**し，二階微分を outer に流す（first-order 近似は
> ablation で比較可）．ANIL なので二階微分のコストは head に対してのみ発生し，
> 軽い．**実装上は train と test で全く同じ inner rule**
> (`geohead.adaptation.test_time.inner_rule_adapt`) を再利用する（`create_graph=True`
> が train，`create_graph=False` が test）ことで，§8.3 の train/test 一貫性要件を
> 満たす．前処理フラグ `preconditioned` は `GeoHeadConfig.preconditioned_inner` / `EvalConfig.inner_preconditioned` から同値に伝搬される（train/test で必ず揃える）．Cholesky solve は微分可能なので outer の二階微分も自然に流れる．

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

## 8. 比較対象（3 学習者 × 4 適応手法）

比較は **2 段の直積構造**：3 種類の学習者 $\{B_1, B_2, P\}$ で得た
$(\theta, \beta_0)$ に対して，4 種類の test-time head 適応
$\{\text{none}, \text{ridge}, \text{geo}, \text{inner}\}$ をすべて組み合わせて評価する．
すなわち **学習者 = encoder と初期 head の作り方**，
**適応 = test 時の head 更新規則**として完全に分離する．

すべての学習者は

- **同じ encoder アーキテクチャ**（§5），
- **同じ training データ条件**（3 training corpora $D_1, D_2, D_3$ のみ使用．
  test corpora $T_1, T_2$ は学習中一切触らない），
- **同じ test-time 評価プロトコル**（§9）

を共有する．**ただし学習中の勾配ステップ数とサンプル消費量は学習者間で同一ではない**
（後述 §8.5）．

### 8.1 学習者 B1: source-only

- **Phase 0 (warm-up)**: 3 corpora をプールした教師あり MSE を
  Adam で `epochs=30, batch_size=256` 最小化する．
  これにより $(\theta, \beta_0)$ の初期点を得る
  (`src/geohead/training/warmup.py::warmup_train`)．
- **Phase 1**: **追加学習なし**．warm-up 終了時点の $(\theta, \beta_0)$ を
  そのまま B1 の最終モデルとする
  (`src/geohead/experiments/sanity.py:492-493`)．
- **役割**: 「meta-learning も DARE-GRAM も無い，純粋な source-only training が
  どこまで届くか」の対照群．

### 8.2 学習者 B2: DARE-GRAM (Nejjar et al. 2023)

- **Phase 0**: B1 と同一の warm-up checkpoint を継承．
- **Phase 1**: episode-based DARE-GRAM training を `outer_steps=5000` 回す
  (`src/geohead/training/baseline.py::baseline_train`)．
  各 step で：
  - 順序対 $(i, j)$ を $\{D_1, D_2, D_3\}$ から $i \ne j$ で一様サンプル
  - source minibatch $B_s \subset D_i$（$|B_s|=64$），
    target minibatch $B_t \subset D_j$（$|B_t|=64$）
  - **supervised MSE は `B_s` のラベルのみ**を使用：
    $L_{\text{src}} = \frac{1}{|B_s|}\sum_{(x,y) \in B_s} (\beta^\top \phi_\theta(x) - y)^2$
  - **DARE-GRAM 整列項**を $Z_s = \phi_\theta(B_s)$，$Z_t = \phi_\theta(B_t)$ から計算：
    - $L_{\cos}\big(\hat G_s^+, \hat G_t^+\big)$（inverse Gram の主成分の cosine 距離）
    - $L_{\text{scale}}\big(\lambda_{\max}\hat G_s^+, \lambda_{\max}\hat G_t^+\big)$（最大固有値の log 比二乗）
  - 最終 loss: $L_{\text{src}} + \alpha_{\cos} L_{\cos} + \gamma_{\text{scale}} L_{\text{scale}}$
    （$\lambda_D$ は掛けず，$\alpha_{\cos}, \gamma_{\text{scale}}$ 自身で重み調整）
- **役割**: 既存の代表的 UDA 回帰手法．
  meta-learning なしで「**特徴の幾何**だけを揃える」ことの効果を測る．

### 8.3 学習者 P: GeoHead（提案手法）

- **Phase 0**: B1, B2 と同一の warm-up checkpoint を継承．
- **Phase 1**: §4 の bilevel ANIL を `outer_steps=5000` 回す
  (`src/geohead/training/geohead.py::geohead_train`)．
  各 step で：
  - 順序対 $(i, j)$ を $i \ne j$ で一様サンプル
  - support $S_i \subset D_i$（$|S_i|=32$），query $Q_j \subset D_j$（$|Q_j|=64$），
    DARE 用 source/target $B_i \subset D_i$（64），$B_j \subset D_j$（64）
  - **inner（仮想 head 更新, 5 step）**:
    `inner_rule_adapt(create_graph=True)` で
    $\beta^* = \beta^*(\theta, \beta_0; S_i)$ を $\theta$ に対して微分可能に展開．
    inner objective は §4.3 の geometry-aware head reg + supervised MSE．
  - **outer（真の更新）**:
    $L_{\text{outer}} = \frac{1}{|Q_j|}\big\|\beta^{*\top} \phi_\theta(Q_j) - y_{Q_j}\big\|^2 + L_{\text{src}}(B_i) + \lambda_D \cdot \text{DARE}(B_i, B_j)$
    を Adam で 1 step．$\theta$ と $\beta_0$ が同時に更新される．
- **Test-time adaptation** との整合: 4 適応手法のうち `inner` は
  **訓練時の inner rule をそのまま** $\beta^* = \text{inner\_rule}(S_t, \beta_0; \theta)$ として
  test 側でも呼ぶ．`create_graph=False` のみが違うので，**train と test の inner 作用素は
  関数として厳密一致**する（テスト
  `tests/training/test_geohead.py::test_geohead_training_inner_rule_matches_test_time_inner_rule`
  で担保）．
- **役割**: support → 仮想 head 更新 → query での真の更新，という
  **bilevel 構造そのもの**が，DARE 単独 (B2) や source-only (B1) を
  超える表現を生むかを検証．

### 8.4 4 つの test-time 適応手法（§4.3 と同じ）

| method | head 更新規則 | $\beta_0$ への依存 | closed-form? |
|---|---|---|---|
| `none`  | $\hat\beta = \beta_0$ | 完全に依存（動かない） | — |
| `ridge` | $(Z_S^\top Z_S + \lambda I)^{-1} Z_S^\top y_S$ | **無し**（$\beta_0$ 無視） | ✓ |
| `geo`   | $\beta_0 + (Z_S^\top Z_S + \lambda H)^{-1} Z_S^\top (y_S - Z_S\beta_0)$ | あり | ✓ |
| `inner` | preconditioned GD 5 step．§4.3 の inner objective | あり（正則化項経由） | ✗ |

ここで $H = \hat\Sigma_S + \varepsilon I$．`inner` のハイパーパラメータ
$(\lambda_h, \eta, K, \varepsilon, \text{preconditioned})$ は GeoHead 学習時の inner と
**完全一致**させて呼ぶ（`EvalConfig` の `inner_*` フィールド）．

### 8.5 学習量の非対称性に関する重要な注意

3 学習者の **勾配ステップ数とサンプル消費量** は厳密には等しくない：

| 項目 | B1 | B2 | P |
|---|---|---|---|
| Phase 0 (warm-up) ステップ | 720 | 720 | 720 |
| Phase 1 (outer) ステップ | **0** | 5,000 | 5,000 |
| 合計勾配ステップ | **720** | 5,720 | 5,720 |
| Phase 1 の 1 step 消費サンプル数 | — | 128 | 224 |
| 合計サンプル消費量 (sample-pass) | 180,000 | 820,000 | 1,300,000 |

つまり「同じ training データ条件」とは「**同じデータプールから引いている**」という
意味であって，「**同じ予算で勾配を回している**」という意味ではない．

この非対称性の影響：

- **B1 は Phase 1 を持たない**が，warm-up 終了時の train MSE が 0.020 程度（noise floor の
  約 8 倍）まで落ちており，**source MSE 単体に関してはほぼ収束**しているため，追加で
  単純 supervised SGD を回しても質的な改善は期待しにくい．
- **B2 と P の outer ステップ数は同一**．唯一の差は 1 step あたりのサンプル数
  （128 vs 224）と損失関数の構造．

この点を厳密に詰めたい場合の ablation は §8.6 に列挙．

### 8.6 追加 ablation（時間が許せば）

| Variant | warm-up | Phase 1 | 内容 | 目的 |
|---|---|---|---|---|
| `B1+` | あり | あり (5000 step pure MSE) | B1 を outer_steps=5000 だけ追加 supervised で回す | B1 の劣位が「ステップ不足」ではなく「source-MSE 単独の本質的限界」であることの検証 |
| `P w/o DARE` | あり | あり | P から $\lambda_D$ 項を 0 に | DARE-GRAM 整列の必要性 |
| `P w/o headreg` | あり | あり | inner 正則化を naive ridge に | geometry-aware reg の必要性 |
| `P first-order` | あり | あり | inner の 2 階微分を切る | 2 階微分の貢献 |
| `P full-MAML` | あり | あり | encoder も inner で更新 | ANIL vs full MAML |
| `P single-source` | $D_1$ のみ | $D_1$ 内 split | 学習コーパス 1 つだけ | multi-source の必要性 |
| `λ_h sweep` | あり | あり | $\lambda_h \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ | head 正則化の効きと B2 の "head norm blow-up" の修復 |
| `inner_steps sweep` | あり | あり | inner $K \in \{1, 3, 5, 10\}$，preconditioned on/off | inner の効きと preconditioner の必要性 |
| `unified-batch` | あり | あり | 1 step あたり A=32, B=64 の 2 batch を 3 学習者で完全共有．予算を厳密に揃える | データ予算厳密一致での fair comparison |

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
| inner | preconditioned | **True**（M3 v5 以降，train / eval 一致が必須） |
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
│   ├── experiments/                     [x]  end-to-end drivers
│   │   ├── __init__.py                  [x]
│   │   └── sanity.py                    [x]  M3 sanity check (B1/B2/P × 4 adapt)
│   └── utils/                           [ ]
│       ├── seed.py                      [ ]
│       └── config.py                    [ ]
├── scripts/                             [x]
│   └── m3_sanity_check.py               [x]  CLI wrapper around experiments.sanity
├── results/                             [ ]  (gitignore; written by scripts/)
└── tests/
    ├── __init__.py                      [x]
    ├── data/                            [x]  test_toy.py, test_episode.py
    ├── models/                          [x]  test_encoder.py, test_head.py
    ├── losses/                          [x]  test_dare_gram.py, test_head_reg.py
    ├── adaptation/                      [x]  test_test_time.py
    ├── training/                        [x]  test_warmup.py, test_baseline.py, test_geohead.py
    ├── evaluation/                      [x]  test_metrics.py, test_runner.py, test_visualize.py
    └── experiments/                     [x]  test_sanity.py
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
- [x] **M3** end-to-end sanity driver (`src/geohead/experiments/sanity.py`, `scripts/m3_sanity_check.py`)
  - 3 学習者 `{B1: source-only, B2: DARE+ridge, P: GeoHead}` が**共通の warm-up
    チェックポイント**から出発し，各々に対して §9.3 の 4-method evaluation
    matrix (`none / ridge / geo / inner`) を走らせる
  - 出力: `records.jsonl`（long-format，`learner` キー付き），`aggregated.csv`
    （seed を潰した mean ± 95 % CI，per `(learner, corpus, k_shot, method)`），
    `plots/`（per-learner と per-method の 2 系統 × `head_correction` scatter），
    `summary.md`（markdown table），`history/{warmup,baseline,geohead}.json`
  - master seed から全副次乱数（toy data, encoder init, warm-up shuffle,
    baseline / geohead episode gen, eval support sub-sampling）を派生させる
    ので同じ config は bit-identical な artefact を再生する
  - `--smoke` フラグで ≈5 s の縮小実行．フル実行は ~15 min（CPU, 1500 outer_steps）

#### M3 の試行ログと最終設定

v1 〜 v4 は **タスク難度と inner rule の stability** を調整した反復．v5 以降は
shift 強度を固定し **preconditioner 周りの $(p, k_{\max}, \varepsilon)$ のトレードオフ**
を探った．

| run | shift (h_train, h_extrap, μ, σ) | $p$ | $k_{\max}$ | $\varepsilon$ | preconditioned | 主要結果 |
|---|---|---|---|---|---|---|
| v1 | (0.4, 0.8, 0.5, 0.1) | 32 | 32 | 1e-6 | ✗ | タスク難度低・`inner` が発散．`B1/none` が noise floor． |
| v2 | (0.8, 1.5, 1.0, 0.1) | 32 | 32 | 1e-6 | ✗ | `inner` 発散消滅．ただし全セルで `none` が最強． |
| v3 | (2.0, 3.0, 1.0, 0.1) | 32 | 32 | 1e-6 | ✗ | `P/none` が `B1/none` を T1 で -18 %．ただし `B1/none` ≈ 0.05（noise floor の 5×）で **source-only で十分賄える regime**． |
| v4 | (2.0, 3.0, 1.0, 0.1) | 32 | 32 | 1e-6 | ✗ (steps=5) | `inner_rule_adapt` がスケール不変でないため 5 step で発散（B2/P）． |
| v5 | (4.0, 6.0, 2.0, 0.05) | 32 | 32 | 1e-4 | ✓ | shift を v3 の倍にして `B1/none T2 = 0.57` まで押し上げ．P/none が T1/T2 両方で最小．ただし `inner` / `geo` が $k = p = 32$ の Marchenko-Pastur エッジで発散（`P/geo k=32 = 7.54`）． |
| v6 | (4.0, 6.0, 2.0, 0.05) | **48** | **24** | **1e-2** | ✓ | $p$ を広げ $k_{\max} < p$ を強制 + ε を damping として増やす．**発散完全解消**．ただし ε=1e-2 が preconditioner を identity に近づけて GeoHead 訓練の inner trajectory が縮み（\|β'-β₀\| = 1.14 → 0.41），P/none の優位性が失われた． |
| v7 | (4.0, 6.0, 2.0, 0.05) | 48 | 24 | **1e-4** | ✓ | ε を v5 相当に戻した．**発散なし**．しかし $p=48$ は $\text{support\_size}=32 < p$ で preconditioner の range が狭く，やはり inner 軌道は短い（\|β'-β₀\| = 0.45）．B2/none が T1 で最強化，P の優位性は戻らず． |
| v8 | (4.0, 6.0, 2.0, 0.05) | **32** | **24** | 1e-4 | ✓ | **採用設定**．$p=32, \text{support\_size}=32$ で訓練 inner を rank-edge（$k=p$）に戻し GeoHead の outer meta-gradient signal を確保（\|β'-β₀\| = 1.14）．同時に eval $k_{\max}=24$ で Marchenko-Pastur 発散を回避．`P/none T1 = 0.073`，`P/none T2 = 0.501`，`P/ridge k=24 T2 = 0.155` がいずれも全セル中最小． |

採用した `_default_toy()` / `_default_geohead()` / `_default_eval()` / `SanityConfig.encoder_p` は v8 に相当する．

#### v5 → v8 の設計原理

v5 で発見した「**訓練時の rank-edge は GeoHead の signal 源，eval 時の rank-edge は数値不安定の源**」という同根の現象を，**train / eval で $k/p$ を非対称に設定**することで両立させた:

- **train 側**: $\text{support\_size} = p = 32$ で rank-edge に居続ける → preconditioner が range を強く歪ませ inner trajectory が長く，outer meta-gradient に強い shape signal が乗る．
- **eval 側**: $k_{\max} = 24 < p = 32$ で Marchenko-Pastur エッジを回避 → $\hat\Sigma_S$ が常に rank-deficient，null space の勾配成分が自然にゼロ，preconditioner が well-behaved．

#### M3 合格根拠（v8）

1. **source-only の構造的不足**: `B1/none T1 = 0.078`（noise floor $\sigma^2 = 0.0025$ の **31×**）．`B1/none T2 = 0.567`（floor の **227×**）．`head_shift=4.0` で 3 corpora の $\beta^*$ をばらけさせたことで pooled 平均ではどの個別 $T$ にもフィットできない．「source で warm-up しただけでは本質的に不足」という claim の実験的裏付け．
2. **meta-learning が β₀ を改善**: `P/none` が T1 (**0.073**)・T2 (**0.501**) の両方で全 `none` 中最小．warm-up だけの $\beta_0$ より GeoHead の meta-trained $\beta_0$ が target-optimal に近い．
3. **adaptation が T2 で劇的に効く**: `P/ridge k=24 T2 = 0.155` は `P/none` 比 **-69 %**，`B1/none` 比 **-73 %**．meta-trained β₀ + ridge head-adapt が最強の combination．
4. **preconditioned inner rule の効果**: v4 の特徴量ノルム依存発散，v5 の $k = p$ エッジ発散はともに $p=32, k_{\max}=24, \varepsilon=10^{-4}$ の v8 設定で解消．Cholesky solve により outer の二階微分も安定に流れる．
5. **adaptation が不要な時は邪魔しない**: T1 では `P/none = 0.073` が `P/ridge k=24 = 0.096` より良い．T1 は interpolation なので $\beta_0$ が既に十分良く，few-shot adapt は variance を足すだけで不要 — これが自然に検出できている．

#### M3 から M4 に引き継ぐ未解決事項

- **`inner` / `geo` の rank-edge 残滓**: $k_{\max}=24$ でも `P/geo k=24 T1 = 0.331` のように $k$ が大きくなるにつれて `inner` / `geo` が緩やかに劣化する（`ridge` は単調減少で劣化なし）．M5 で shrinkage 系（Ledoit-Wolf）や adaptive $\varepsilon$ を ablation．
- **B2 / P が T1 で `none` > `ridge`**: T1 は interpolation で $\beta_0$ が既に target-optimal に近いので，adaptation の variance が bias 削減を上回る．これは期待される挙動だが，`ridge_lambda` をより攻撃的に下げる余地はある．M4 では λ sweep を small set でも入れる．
- **outer loss の重み**: `λ_D, α_cos, γ_scale` は design doc §10 の初期値のまま．`head_shift=4.0` が強いので DARE regularizer が `L_qry` と競合する可能性あり．M4 で小規模 λ_D sweep か M5 の ablation に回す．

### M4: 実験本番（full-variance sample efficiency）
- [x] **M4** main driver (`src/geohead/experiments/main.py`, `scripts/m4_main.py`)
  - **目的**: M3 v8 で確立した設計を拡張し，**データ実現・モデル初期化・訓練確率性・eval サブサンプル**の 4 種の変動を同時に吸収した 95 % CI 付き sample efficiency curve を算出する（domain-generalization 論文の full-variance protocol, cf. Gulrajani & Lopez-Paz 2021）．
  - **設定**（`M4Config` 既定値）:

    | 項目 | M3 v8 | **M4** |
    |---|---|---|
    | learners | B1, B2, P | 同じ |
    | test corpora | T1, T2 | 同じ |
    | methods | none, ridge, geo, inner | 同じ |
    | `k_shots` | (1, 4, 8, 16, 24) | 同じ |
    | toy shifts / `encoder_p` / ε / preconditioned | v8 採用値 | 同じ |
    | `baseline.outer_steps` | 1 500 | **5 000** |
    | `geohead.outer_steps` | 1 500 | **5 000** |
    | `eval.n_seeds` | 5 | **20** |
    | **`n_train_seeds`** | 1（暗黙） | **3** |
    | 1 セル当たりのサンプル数 | 5 | **60** = $3 \times 20$ |

  - **seed 派生**（完全独立な訓練）: train 実行 $i = 0, 1, 2$ に対して `master_seed_i = master_seed + i \cdot 10^6`．この差分は `_run_pipeline_once` 内部で派生する副次 seed（`+1, +2, +3, +4, +1000..+1019`）の全てと衝突しない．各 $i$ で toy dataset・encoder init・warm-up shuffle・訓練 RNG・eval sub-sample の全てが独立．
  - **集計**: `records.jsonl` は `train_seed` 列を持つ 7 200 行．`aggregate()` は `(corpus, k_shot, method)` で group-by するので `train_seed` は自然に収束され，各セルの CI は $n_{\text{train}} \cdot n_{\text{eval}} = 60$ 標本ベース．
  - **出力物**: `results/m4_main/` 配下に `config.json`，`records.jsonl`，`aggregated.csv`，`plots/*.png`，`summary.md`，および `run_{0,1,2}/config.json, run_{0,1,2}/history/*.json`．
  - **実行時間見積もり**（GPU, v8 の 55 s ベース）: `outer_steps` が 1 500 → 5 000 で $\times 3.3$，`n_train_seeds` が 1 → 3 で $\times 3$，合計 $\approx 9 \cdot 55 \text{ s} \approx 8$ 分．`--smoke` は $n_{\text{train}} = 2, \text{outer\_steps} = 50, n_{\text{eval}} = 2$ で $\approx 30$ s．
  - **CLI**:
    - `python -m scripts.m4_main --device cuda` で full run．
    - `python -m scripts.m4_main --smoke` で end-to-end smoke．
    - `--n-train-seeds`, `--n-seeds`, `--baseline-outer-steps`, `--geohead-outer-steps` で schedule 上書き可．

#### M4 v2: 軸統一 + 比較表機能 (`results/m4_main_v2/`)

v1 と完全同一の seed・hyperparams（`master_seed=0, n_train_seeds=3, eval.n_seeds=20, outer_steps=5000`）で再実行，**1 セル当たり 60 標本**の 95 % CI 付き．差分は post-processing のみ:

- **per-corpus で全 plot の y 軸を log-scale に統一**：`_compute_unified_axes` が `(min(mean-CI)*0.85, max(mean+CI)*1.25)` を計算し，`sample_efficiency_{corpus}_{learner}.png` と `*_by_{method}.png` の全てに適用．これにより P／B1／B2 の PNG を並べた時に同スケールで対比可能．
- **`comparison.md` 出力**：5 セクション構成で paper-ready な比較資料として独立．headline winner per `(corpus, k)`，P-vs-B1／B2 の相対改善率，learner 内の method ranking（Borda），bold-best 付き完全 MSE 表，head-correction sanity（`‖β̂−β₀‖₂ / Δ_geo` 比）．

#### M4 v2 で確定した 4 つの結論

**結論 1：GeoHead (P) は warm-up として全方位で最強**

T2（外挿）の最大 $k=24$ で 95 % CI が完全に重ならない明瞭な順序：

- **P/ridge = 0.181 ± 0.013**
- B1/ridge = 0.256 ± 0.027 → **P が −29.4 %**
- B2/ridge = 0.301 ± 0.046 → **P が −39.9 %**

T1（内挿）でも `P/none = 0.0652 ± 0.0035` が `B1/none = 0.0910 ± 0.0062` を −28.4 % 下回り，どの test-time 手法に対しても P 由来の特徴が最も効率よく head 補正できる．**meta-trained $\beta_0$ が pooled supervised $\beta_0$ より target-optimal に近い**ことの 60 標本ベースの直接的な実験的裏付け．

**結論 2：最適 adapt 手法は $k$ に応じて切り替わる**

| corpus | $k \le 4$ | $k = 8$ | $k \ge 16$ |
|---|---|---|---|
| T1 | P / `none` または P / `inner` | P / `inner` | **P / `none`**（P / `ridge` も同等） |
| T2 | P / `inner` | **P / `ridge`** | **P / `ridge`** |

- 外挿（T2）では $k$ が増えるほど closed-form ridge が圧勝．meta-trained $\beta_0$ から ridge 補正で bias-variance を最適化する経路が最速．
- 内挿（T1）では meta-trained $\beta_0$ がほぼ既に target-optimal で，下手にいじると variance が bias 削減を上回る → `none` が安全な選択．これは「meta-init の質」を端的に示す．

**結論 3：DARE-GRAM (B2) の "head norm blow-up" を定量化**

$k=24$ における $\| \hat\beta - \beta_0 \|_2 / \Delta_{\text{geo}}$ 比（`comparison.md §5`）:

| learner | corpus | `geo` | `inner` | `ridge` |
|---|---|---|---|---|
| P | T1 | 12.93 | 18.07 | **5.29** |
| P | T2 | 5.85 | 8.06 | **1.64** |
| B1 | T1 | 13.92 | 19.49 | 6.40 |
| B1 | T2 | 5.92 | 8.23 | 2.20 |
| **B2** | T1 | **103.07** | **113.52** | 5.08 |
| **B2** | T2 | **45.05** | **45.29** | 1.96 |

B2 は `geo` / `inner` で **P や B1 の 5–10 倍**の比率を示す．これは「$\Delta_{\text{geo}}$ は同程度なのに $\| \hat\beta - \beta_0 \|_2$ だけが暴れている」状態，すなわち **$\hat\Sigma_S$ の null/低固有値方向に head を伸ばしているが，Q-error には寄与しない** 失敗モード．DARE で source/target の Gram を揃える副作用で feature scale が重要でない方向に膨張した結果と解釈でき，**geometry-aware head 正則化の不在が few-shot 体制で具体的にどう壊れるか**を可視化する重要な観察．

**結論 4：`ridge` が最も robust，`geo` は $k$ 大で崩壊**

method ranking（learner 内 Borda 平均，n=10 セル）：

- P : `inner` 1.80 < `ridge` 2.30 < `none` 2.70 < `geo` 3.20
- B1: `inner` 1.90 < `ridge` 2.00 < `geo` 3.00 < `none` 3.10
- B2: `ridge` 2.00 < `none` = `inner` 2.20 < `geo` 3.60

- **`ridge`** は全 learner / 全 $k$ で単調 monotone-decreasing，B2/ridge T2 ですら $k=4 \to 24$ で $0.65 \to 0.30$ と素直に効く．**論文では deployable な default 推奨手法**．
- **`geo`** は $k$ 大で逆に劣化（P/`geo` T1 $k=24 = 0.176$ vs $k=8 = 0.077$）．closed-form geometry-aware は $\hat\Sigma_S$ 推定誤差に敏感で，$k$ が増えると bias/variance の balance が崩れる．M5 で shrinkage（Ledoit-Wolf 等）で改善余地あり．
- **`inner`** は $k \le 8$ で勝つが $k \ge 16$ では `ridge` に逆転される（`P/inner k=24 T2 = 0.339` vs `P/ridge = 0.181`）．`inner_steps=5` の有限ステップが $k$ 大で過小に．M5 で `inner_steps` sweep．

#### M4 から M5 に引き継ぐ ablation 候補

1. **head 正則化 $\lambda_h$ sweep**：結論 3 の "head norm blow-up" を `B2 + head-reg` で抑えられるかを直接検証．論文の **geometry-aware head reg の必要性主張** を切り分ける核心実験．
2. **preconditioned inner × `inner_steps` sweep**：結論 4 の `inner` の $k$ 大での劣化を `inner_steps ∈ {1, 3, 5, 10}` で改善できるか，preconditioner の寄与を非対称 ablation で測る．
3. **DARE 重み $\alpha_{\cos}, \gamma_{\text{scale}}$ sweep**：`B2 vs B1` の差は本質的に DARE 効果のみ．T2 で `B2 > B1`（悪化）になっているので **DARE は T2 への外挿には寄与しない or 害**という強い反論材料になる可能性．
4. **shrinkage 系 $\hat\Sigma_S$ 推定**：結論 4 の `geo` 崩壊の修復候補．Ledoit-Wolf や adaptive $\varepsilon$ で `geo` を `ridge` 並に robust にできるか．

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
