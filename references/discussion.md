# 引き継ぎ用詳細メモ

## 0. この研究で本当に問題にしたいこと

この研究の中心問題は，単なる domain adaptation regression の性能改善ではない．  
本質的に考えたいのは，

> **少数のターゲットラベルから，ターゲット側の条件付き予測関数，特に \(E_t[Y\mid Z]\) をどう復元するか**

という点である．

ここで，

- \(X\): 元の入力
- \(Z=\phi_\theta(X)\): 深層エンコーダが出す表現
- \(Y\): 連続値ラベル
- source / target 間で covariate shift も conditional shift もありうる

という前提で考えている．

重要なのは，最終目標を

- 「source と target で shared regressor を成立させること」

には置いていないことである．  
むしろ，**条件付き分布シフトを積極的に考えるなら，shared regressor は一般には成立しない**と考えている．

したがって，目標は

> **source で得た head / predictor 初期値から，few-shot target labels により target-optimal predictor に素早く到達できるような表現と適応則を学ぶこと**

である．

---

## 1. Chen と DARE-GRAM をどう参照しているか

### 1.1 Chen 系（RSD 系）をどう見ているか

Chen 系の重要な点は，

- domain adaptation for regression では
- classification 的な単純な feature alignment がうまくいかない
- 特に feature scale の扱いが重要

という問題提起をしている点である．

こちらはこの問題提起自体は強く支持している．  
つまり，

- 「source / target の表現をただ近づければ良い」
- 「分類で使われる alignment をそのまま流用すれば良い」

とは考えていない．

ただし，Chen 系を参照しつつ，こちらの問題意識はそこからさらに一歩進んでいる．  
つまり，

> regression では単なる feature alignment では足りない  
> では，**何が target risk の悪化を支配しているのか**  
> それを finite-dimensional・線形代数的に書けないか

という方向に進めたい．

---

### 1.2 DARE-GRAM をどう見ているか

アップロードされた DARE-GRAM 論文の要点は次である．

- deep encoder \(h_\theta\) の後ろに linear regressor \(g_\beta\) がつく構造を考える
- 従来は \(Z_s, Z_t\) の feature alignment に寄りがちだが，それでは不十分
- OLS 解
  \[
  \hat\beta=(Z^\top Z)^{-1}Z^\top Y
  \]
  に現れる inverse Gram に注目し，
- source / target の pseudo-inverse Gram の角度とスケールを整合させることで，回帰 adaptation を改善する

という方法である :contentReference[oaicite:0]{index=0}

DARE-GRAM の total loss は

\[
L_{\text{total}}
=
L_{\text{src}}
+\alpha_{\cos}L_{\cos}
+\gamma_{\text{scale}}L_{\text{scale}}
\]

である :contentReference[oaicite:1]{index=1}

ここで

- \(L_{\text{src}}\): source 回帰 MSE
- \(L_{\cos}\): pseudo-inverse Gram の角度整合
- \(L_{\text{scale}}\): 主固有値のスケール整合

である．

さらに，DARE-GRAM の encoder は特別な adversarial encoder ではなく，論文では **ImageNet 事前学習済み ResNet-18** を backbone / encoder として用いている :contentReference[oaicite:2]{index=2}  
つまり DARE-GRAM は **adversarial learning ベースではない**．  
関連研究として adversarial DA は触れられているが，提案手法自体は source supervised loss + inverse-Gram regularization を end-to-end に足し込むだけである 

---

### 1.3 DARE-GRAM を今回どう使いたいか

ただし，こちらの設定では DARE-GRAM を**原論文通りの意味**では使わない．  
原論文はかなり素朴には

> source / target で shared linear regressor が成立しやすい表現を作る

という方向に見える．

しかし，こちらは **conditional shift を積極的に考える**ので，

\[
P_s(Y\mid Z)\neq P_t(Y\mid Z)
\]

を前提にしている．  
したがって，

- source / target で最終的に shared regressor が成立する

という仮定には乗らない．

ここでの DARE-GRAM の位置づけは，むしろ

> **shared regressor を成立させること**ではなく，  
> **source-derived head を target head に few-shot で補正しやすいような表現幾何を作る regularizer**

である．

つまり DARE-GRAM loss は，

- conditional shift を消す loss
ではなく，
- conditional shift が残っていても few-shot head adaptation が効きやすい表現を作る loss

として使いたい．

---

## 2. こちらが感じている「条件付き分布シフト」の問題点

こちらの問題意識の核は，

> regression DA においては，source と target で最適 predictor / 最適 head が違うのではないか

という点にある．

より正確には，表現 \(Z\) を固定したとき，

\[
P_s(Y\mid Z)\neq P_t(Y\mid Z)
\]

なら，一般に source で最適な linear head \(\beta_s^\star\) と target で最適な linear head \(\beta_t^\star\) は一致しない．

このとき，

- feature alignment がうまくいっても
- source head をそのまま target に持ち込むだけでは
- target risk が悪化する

可能性がある．

したがって，問題の本質は

> **feature alignment の是非**だけではなく，  
> **source-optimal head と target-optimal head のズレが，target risk をどう悪化させるか**

であると見ている．

さらに，few-shot setting では，

- source は豊富
- target は少数ラベルしかない

ので，欲しいのは

> **source head を捨てて target head を一から学ぶこと**

ではなく，

> **source head を初期値・参照点として，few-shot target labels で target head へ少し補正すること**

である．

つまり条件付き分布シフトを考えると，shared regressor は最終目標ではなく，  
**source-derived initialization** として扱うべきだ，という立場である．

---

## 3. こちらで導出した式とその意味

この部分がかなり重要．  
すでに別チャット引き継ぎ用に整理された詳細要約があり，そこでもかなり詰めている :contentReference[oaicite:4]{index=4}

以下，エッセンスをまとめる．

---

### 3.1 基本設定

target risk を

\[
R_t(\beta)=\mathbb E_t[(Y-Z^\top\beta)^2]
\]

と置く．  
ここで \(\mathbb E_t\) は target 分布上の期待値．  
\(Z\) は target domain 上の表現である．

target conditional mean を

\[
m_t(Z)=\mathbb E_t[Y\mid Z]
\]

とすると，恒等的に

\[
R_t(\beta)
=
\mathbb E_t[(Y-m_t(Z))^2]
+
\mathbb E_t[(m_t(Z)-Z^\top\beta)^2]
\]

と分解できる．

これは，

- 第1項: 表現 \(Z\) を固定したときの不可避ノイズ
- 第2項: その表現上で線形 predictor が conditional mean に届いていない誤差

という意味である :contentReference[oaicite:5]{index=5}

---

### 3.2 target で線形 conditional mean を仮定した場合

さらに

\[
m_t(Z)=Z^\top\beta_t^\star
\]

を仮定すると，

\[
Y=Z^\top\beta_t^\star+\varepsilon_t,\qquad \mathbb E_t[\varepsilon_t\mid Z]=0
\]

なので，

\[
R_t(\beta)
=
\mathbb E_t[\varepsilon_t^2]
+
(\beta_t^\star-\beta)^\top \Sigma_t (\beta_t^\star-\beta)
\]

となる．  
ここで

\[
\Sigma_t=\mathbb E_t[ZZ^\top]
\]

である．  
これは厳密には二次モーメント行列であり，共分散行列ではないことに注意する必要がある :contentReference[oaicite:6]{index=6}

特に source-optimal head \(\beta_s^\star\) を target に持ち込むと，

\[
R_t(\beta_s^\star)-R_t(\beta_t^\star)
=
(\beta_s^\star-\beta_t^\star)^\top \Sigma_t (\beta_s^\star-\beta_t^\star)
\]

となる．

この式の意味は，

- head mismatch が大きいほど悪い
- ただしそのズレの影響は target 側の geometry \(\Sigma_t\) により重み付けされる
- すなわち，ただ head の差だけを見ても足りず，**target 上でその差がどれだけ意味を持つか**が重要

ということ．

これは

\[
\delta^\top\Sigma_t\delta
=
\mathbb E_t[(Z^\top\delta)^2]
\]

とも読めるので，  
**source head と target head の差が，target 上で予測値をどれだけ動かすかの平均二乗**と解釈できる :contentReference[oaicite:7]{index=7}

---

### 3.3 この式の価値

この式単独は，数学的には自然で，自明に見えやすい．  
したがって，単独の新規理論として押し出すのは弱い．

しかし価値は，

> **shared regressor 仮定が壊れたとき，何が target risk を悪化させるかを finite-dimensional に診断する座標系**

を与えることにある．

つまり，これは研究の終点ではなく，**その後の方法論設計の出発点**である :contentReference[oaicite:8]{index=8}

---

### 3.4 head mismatch の分解

母集団線形回帰解

\[
\beta_d^\star=\Sigma_d^{-1}c_d,\qquad
\Sigma_d=\mathbb E_d[ZZ^\top],\quad
c_d=\mathbb E_d[ZY]
\]

を使うと，

\[
\beta_s^\star-\beta_t^\star
=
\Sigma_s^{-1}c_s-\Sigma_t^{-1}c_t
\]

であり，

\[
\beta_s^\star-\beta_t^\star
=
(\Sigma_s^{-1}-\Sigma_t^{-1})c_s+\Sigma_t^{-1}(c_s-c_t)
\]

と分解できる．

これにより head mismatch は

- **geometry shift**
  \[
  (\Sigma_s^{-1}-\Sigma_t^{-1})c_s
  \]
- **feature-label coupling shift**
  \[
  \Sigma_t^{-1}(c_s-c_t)
  \]

に分けて読めることになる :contentReference[oaicite:9]{index=9}

この分解をさらに risk に代入すると，

\[
\delta=A+B
\]

として

\[
R_t(\beta_s^\star)-R_t(\beta_t^\star)
=
A^\top\Sigma_tA + 2A^\top\Sigma_tB + B^\top\Sigma_tB
\]

となり，

- geometry term
- coupling term
- interaction term

に分解できる．

この点はかなり重要で，  
**geometry だけ合わせればよいわけではなく，feature-label coupling も問題である**  
という認識につながっている :contentReference[oaicite:10]{index=10}

---

### 3.5 few-shot adaptation との接続

この式から自然に出る考えは，

- source で得た head \(\beta_s\) がある
- target で少数 support が来る
- その support から \(\beta_s\) を \(\beta_t\) に少し補正する
- 補正量が小さくて済むほど良い表現である

というものである．

つまり few-shot adaptation は，

> **target few-shot labels によって，source-target head mismatch をどれだけ補正できるか**

という問題として読むべきだ，という立場である :contentReference[oaicite:11]{index=11}

---

## 4. MAML / bilevel 的にどう捉えているか

こちらの MAML 理解は次の通り．

- MAML は「良い初期値を学ぶ機構」である
- inner では support を使って仮想的に更新する
- outer では，その更新後パラメータが query で良い性能を出すように，元の初期値を更新する
- 結果として，「少数 support でうまく適応できる初期値」を学ぶ

この理解は概ね正しい，という整理になっている．  
ただし今回やりたいのは，MAML をそのまま使うことではなく，

> **few-shot support で head を補正しやすい representation と初期 head を outer で学ぶ bilevel 学習**

である．

また，support / query の使い方については，

- inner に support
- outer に query

が自然であり，  
逆に inner=query, outer=support は，MAML の本来の few-shot adaptation の意味からは外れるので，基本的には非推奨，という整理になっている．

---

## 5. 今考えている最終的な方法論の骨格

### 5.1 全体方針

現時点で考えているのは，

- **DARE-GRAM のロスは必ず用いる**
- ただしそれだけではなく，
- **few-shot support による head adaptation を明示的に入れる**
- その際，head の補正量は target/task geometry で重み付けする
- これを MAML / bilevel 風に組む

という方針である．

---

### 5.2 DARE-GRAM loss の位置づけ

今回の方法では，DARE-GRAM loss は **outer 側**に置くのが自然という結論になっている．  
理由は，

- DARE-GRAM は representation geometry を整える loss
- few-shot support での head correction を定めるものではない

からである．

したがって，inner には DARE-GRAM を置かず，  
**outer で encoder にかける regularizer**として用いる．

---

### 5.3 inner で使う loss

inner では，support 上で head \(\beta\) を更新する．  
その loss は

\[
L_{\text{inner}}
=
L_{\text{sup-pred}}
+
\lambda_h L_{\text{head-reg}}
\]

であり，

\[
L_{\text{sup-pred}}
=
\sum_{(x,y)\in S_d}(y-\beta^\top \phi_\theta(x))^2
\]

\[
L_{\text{head-reg}}
=
(\beta-\beta_0)^\top(\hat\Sigma_d+\varepsilon I)(\beta-\beta_0)
\]

のような geometry-aware head regularization を考えている．  
ここで \(\beta_0\) は source-derived な初期 head / meta-initial head であり，  
「shared optimal regressor」ではなく **adaptation の出発点**である．

---

### 5.4 outer で使う loss

outer では，

- query prediction loss
- DARE-GRAM loss

を組み合わせる．

\[
L_{\text{outer}}^{(d)}
=
L_{\text{qry}}^{(d)}
+
\lambda_D
\Big(
L_{\text{src}}^{(d)}
+\alpha_{\cos}L_{\cos}^{(d)}
+\gamma_{\text{scale}}L_{\text{scale}}^{(d)}
\Big)
\]

ここで

\[
L_{\text{qry}}^{(d)}
=
\sum_{(x,y)\in Q_d}(y-(\beta_d')^\top \phi_\theta(x))^2
\]

である．

解釈としては，

- inner: 少数 support から head をどう補正するか
- outer: その補正後に query でうまくいくような encoder / initial head を学ぶ
- DARE-GRAM: その outer で，few-shot support からの補正が効きやすい geometry を作る

となる．

---

## 6. toy 実験の基本的な考え方

### 6.1 toy で何を検証したいか

toy で見たい主張は少なくとも次の2点．

1. **DARE-GRAM 的 geometry regularization だけでは不十分**
2. **few-shot head adaptation だけでも不十分**
3. **両者を組み合わせると，少数 support から target predictor をより効率よく回復できる**

である．

つまり，  
**表現幾何と head correction cost の両方を尊重した最適化**を検証したい．

---

### 6.2 toy データの構造

最終的に，toy の設定は

> **x の分布は同じ**という制約は捨てる

ことになった．  
むしろ最初から，

\[
x \sim \mathcal N(\mu_d,\Sigma_d)
\]

で domain ごとに covariate 分布を変える．

さらに共通の真の非線形表現

\[
z^\star=\phi^\star(x)
\]

を通し，

\[
y=(\beta_d^\star)^\top z^\star+\varepsilon
\]

とする．

ここで，

- \((\mu_d,\Sigma_d)\) の違い → geometry / covariate shift
- \(\beta_d^\star\) の違い → conditional mean shift / head shift

が同時に入る．

つまり toy は最初から，

\[
p_s(x)\neq p_t(x),\qquad P_s(Y\mid Z)\neq P_t(Y\mid Z)
\]

を持つ設定にする．

この方が，今回の主張と整合的である．

---

### 6.3 モデル

toy では DARE-GRAM 論文の ResNet-18 を使う必要はなく，  
小さな MLP encoder で十分である．

構造は

\[
x \xrightarrow{\phi_\theta\ \text{(small MLP)}} z \xrightarrow{\beta^\top z} \hat y
\]

でよい．

重要なのは encoder の種類ではなく，  
**encoder の出力特徴 \(Z\) に対して DARE-GRAM loss と head-adaptation loss を掛けること**である．

---

## 7. 実験の比較対象

ここは何度か議論して，最終的に重要な修正が入っている．

### 7.1 最初の案

最初は

- DARE-GRAM
- 提案法

の2本比較でもよいのでは，という話があった．

しかしその後，

> **DARE-GRAM 側も few-shot adaptation にしないと比較が不公平**

という結論になった．

理由は，提案法だけが target support を使えて，DARE-GRAM が使えないなら，差が出ても「few-shot support を使ったかどうか」の差になってしまうからである．

---

### 7.2 現在の理想比較

最もきれいなのは次の3本比較である．

#### Baseline 1: DARE-GRAM + naive few-shot
- 学習時: DARE-GRAM loss のみ
- テスト時: target support で **単純 ridge / OLS / head-only fine-tuning**

例：
\[
\hat\beta_t^{\text{DARE}}
=
\arg\min_\beta
\sum_{(x,y)\in S_t}(y-\beta^\top \phi_\theta(x))^2
+
\lambda \|\beta-\beta_0\|_2^2
\]

#### Baseline 2: DARE-GRAM + geometry-aware few-shot
- 学習時: DARE-GRAM loss のみ
- テスト時: target support で **geometry-aware head adaptation**

例：
\[
\hat\beta_t
=
\arg\min_\beta
\sum_{(x,y)\in S_t}(y-\beta^\top \phi_\theta(x))^2
+
\lambda(\beta-\beta_0)^\top(\hat\Sigma_t+\epsilon I)(\beta-\beta_0)
\]

#### Proposed: full proposed
- 学習時: DARE-GRAM + bilevel / meta objective
- テスト時: geometry-aware few-shot adaptation

この3本比較にすると，

- test-time adaptation rule の効果
- train-time bilevel/meta-learning の追加効果

を分離して見られる．

---

## 8. toy の評価設計

### 8.1 test set を複数用意するとはどういうことか

単に random split を複数作るというより，  
**複数の target domain / target shift condition** を用意する，という意味である．

たとえば，

- moderate head shift
- strong head shift
- geometry shift + moderate head shift
- geometry shift + strong head shift

のような複数 target 条件を作り，  
それぞれで評価する．

これにより，

> 「特定の1個の target でたまたま効いた」

ではなく，

> **複数の target shift 条件で一貫して few-shot adaptation 性能が良いか**

を見られる．

---

### 8.2 support size

各 target condition について，

\[
k \in \{1,3,5,10,20\}
\]

などで support size を振る．

そして各 \(k\) について，query MAE / MSE を比較する．  
見たいのは，

> **極小の support で提案法の方がより sample-efficient に target query risk を下げられるか**

である．

---

### 8.3 指標

主指標は

- target query MSE
- target query MAE

でよい．  
理論との接続は MSE の方が強いが，MAE も併記してよい．

さらに補助指標として，

- adaptation 前後の query error
- head correction size
  \[
  \|\beta'-\beta_0\|,\qquad
  (\beta'-\beta_0)^\top \hat\Sigma_t (\beta'-\beta_0)
  \]
- inverse Gram / Gram の可視化

も見るべきである．

head correction size は，こちらの理論式との接続上かなり重要である．

---

## 9. この研究の主張として今置くべきもの

ここまでを踏まえると，今の主張は次のように整理される．

### 避けるべき言い方
- source / target で shared regressor を成立させる
- conditional shift を消して共通 predictor を学ぶ

これは現在の立場とは整合しない．

### 使うべき言い方
- source-derived head を target few-shot labels で迅速に補正可能な表現を学ぶ
- inverse-Gram geometry を整えることで，few-shot head adaptation の sample efficiency を改善する
- 表現 geometry と head correction cost を jointly 考慮する

つまり，DARE-GRAM は**共通 head の成立**のためではなく，  
**few-shot recoverability を高める representation regularizer**として再解釈されている．

---

## 10. 現時点での実験ストーリーの最終形

1. **Chen / DARE-GRAM の問題意識を参照する**  
   regression DA では単純な feature alignment は危険であり，linear regressor / OLS / Gram 構造に注意すべきである．

2. **しかし，条件付き分布シフトを考えるなら，shared regressor は一般には成立しない**  
   source-optimal head と target-optimal head はズレる．

3. **そのズレがどう risk を悪化させるかを finite-dimensional に書く**  
   \[
   R_t(\beta_s^\star)-R_t(\beta_t^\star)
   =
   (\beta_s^\star-\beta_t^\star)^\top \Sigma_t (\beta_s^\star-\beta_t^\star)
   \]
   さらに geometry shift / coupling shift に分解する．

4. **few-shot adaptation の本質を，この mismatch の補正とみなす**  
   support によって source head から target head に少し補正する．

5. **方法論としては，outer に DARE-GRAM，inner に geometry-aware head adaptation を置く**  
   DARE-GRAM は shared regressor を成立させるためではなく，few-shot head correction が効きやすい表現を作る regularizer とみなす．

6. **toy 実験では，covariate shift と conditional shift の両方を持つ複数 domain を作る**  
   それぞれで few-shot support / query に分け，  
   DARE-GRAM + naive FS，DARE-GRAM + geometry-aware FS，full proposed を比較する．

---

## 11. 補足：すでに別途整理されている長文要約

5-1 の理論確認，head mismatch，geometry/coupling 分解，few-shot adaptation との接続，loss 設計整理については，すでにかなり詳細な引き継ぎ文が生成されている．  
今回の内容はそれをさらに進めて，

- DARE-GRAM 論文の具体的役割
- shared regressor を最終目標にしないという修正
- fair な比較のために DARE-GRAM 側も few-shot adaptation を入れる必要性
- toy 実験の target condition / support size / comparison protocol

まで明確化したものと考えてよい :contentReference[oaicite:12]{index=12}