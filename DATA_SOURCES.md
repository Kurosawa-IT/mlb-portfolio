# データ出典・取得・公開方針

確認日：2026-10-31。コードのライセンスとデータの利用権は別である。

- Statcast / Baseball Savant: https://baseballsavant.mlb.com/csv-docs （列定義）、https://baseballsavant.mlb.com/statcast_search （検索・CSV）。球速・打球速度はmph、plate_x/plate_zはft、角度は度。plate_xは捕手視点で扱い、投打左右別に比較する。
- pybaseball: https://github.com/jldbc/pybaseball 、https://github.com/jldbc/pybaseball/blob/master/docs/statcast.md 、https://github.com/jldbc/pybaseball/blob/master/docs/pitching_stats.md 。バージョン2.2.7を使用。
- FanGraphs pitching leaderboards: https://www.fangraphs.com/leaders/major-league?pos=all&stats=pit&lg=all&qual=0 。2018–2025年、qual=0、チーム通算の投手シーズン成績。旧HTML経路の空応答を検知し、同サービスのランキング用JSONへフォールバックする。URL・時刻・取得件数・ファイルSHA256は reports/acquisition.jsonl に保存。
- MLB利用条件: https://www.mlb.com/official-information/terms-of-use
- FanGraphs利用条件: https://www.fangraphs.com/about/terms-of-service

公開データであることを、自由な再配布の許諾と同一視しない。上記利用条件を確認し、本プロジェクトでは生データの再配布を行わない。公開する対象は分析コード、Notebookの集計・図、評価結果。データそのものへの独立のオープンライセンスは確認できていない。利用目的や公開範囲を拡大する場合は出典の最新条件を再確認し、必要な許諾を得る。

data/raw は .gitignore で除外。取得済みParquetをローカルで再利用する。キャッシュを消すと再取得され、後日のデータ訂正により結果が変わり得る。Notebookには選手行や全投球データの大量出力を埋め込まない。ファイルハッシュは同じスナップショットを使ったかの照合用であり、第三者へのデータ配布ではない。

なおFanGraphs APIは正式な安定性保証付きAPIとして扱わない。ページ構造・列・アクセス方法が変更された場合は取得が停止する。アクセス制限や認証の回避は実装していない。
