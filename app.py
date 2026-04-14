"""
PDF・画像領収書 → マネーフォワード クラウド会計 仕訳インポートCSV 変換ツール
Gemini API 対応版 / SharePoint自動アップロード対応
==========================================================================
"""

import streamlit as st
import pdfplumber
import pandas as pd
import urllib.request
import json
import io
import base64
import requests
import fitz  # PyMuPDF
import time
from PIL import Image
from datetime import datetime

# ============================
# マネーフォワード CSV 設定
# ============================
MF_COLUMNS = [
    "取引日", "決算整理", "借方勘定科目", "借方補助科目", "借方部門",
    "借方税区分", "借方金額", "借方税額", "貸方勘定科目", "貸方補助科目",
    "貸方部門", "貸方税区分", "貸方金額", "貸方税額", "摘要", "仕訳メモ", "タグ", "MF仕訳ID"
]

DEBIT_ACCOUNTS = [
    "旅費交通費", "接待交際費", "消耗品費", "通信費", "広告宣伝費",
    "水道光熱費", "地代家賃", "修繕費", "雑費", "仕入高",
    "外注費", "支払手数料", "その他"
]
CREDIT_ACCOUNTS = [
    "現金", "普通預金", "当座預金", "未払金", "クレジットカード",
    "事業主借", "役員借入金", "短期借入金", "買掛金", "その他"
]
TAX_CATEGORIES = [
    "課税仕入10%", "課税仕入8%（軽減）", "課税仕入8%", "非課税仕入", "不課税"
]

SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
MIME_MAP = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

# ============================
# API キーの読み込み
# ============================
def get_gemini_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

# ============================
# Gemini API プロンプト
# ============================
EXTRACTION_PROMPT = """この領収書・レシート・請求書の画像またはテキストから、日本語の会計仕訳に必要な情報をすべて抽出し、以下のJSON形式のみで返してください。
説明文やコードブロックは不要です。JSONだけ返してください。

{
  "取引日": "YYYY/MM/DD形式。不明な場合は今日の日付",
  "支払先": "支払先・店名",
  "摘要": "内容の簡潔な説明（例：〇〇社 接待費、〇〇駅 交通費）",
  "税込合計金額": 数値のみ（例：11000）,
  "消費税額": 数値のみ（例：1000）,
  "推奨借方勘定科目": "旅費交通費/接待交際費/消耗品費/通信費/広告宣伝費/水道光熱費/地代家賃/修繕費/雑費/仕入高/外注費/支払手数料/その他 のいずれか",
  "推奨貸方勘定科目": "現金/普通預金/未払金/クレジットカード のいずれか",
  "税区分": "課税仕入10%/課税仕入8%（軽減）/課税仕入8%/非課税仕入/不課税 のいずれか",
  "備考": "特記事項があれば記載、なければ空文字"
}
"""

# ============================
# ファイル処理
# ============================
def get_file_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower()

def render_pdf_as_image(pdf_bytes: bytes, page_num: int = 0) -> bytes:
    """PDFの指定ページを画像（PNG）として返す"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]
    mat = fitz.Matrix(2.0, 2.0)  # 2倍ズームで高画質
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")

def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        text = f"[PDFテキスト抽出エラー: {e}]"
    return text.strip()

def call_gemini_api(api_key: str, parts: list, max_retries: int = 3) -> dict:
    """Gemini REST APIを呼び出してJSONを返す（503/429時は指数バックオフでリトライ）"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = json.dumps(
        {"contents": [{"parts": parts}]},
        ensure_ascii=False
    ).encode("utf-8")

    last_error = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            # 成功したらループを抜ける
            break
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            last_error = f"HTTP {e.code}: {error_body}"
            # 503（過負荷）・429（レート制限）はリトライ対象
            if e.code in (503, 429) and attempt < max_retries - 1:
                wait_sec = 2 ** attempt  # 1回目:1秒, 2回目:2秒, 3回目:4秒
                time.sleep(wait_sec)
                continue
            raise Exception(last_error)
    else:
        raise Exception(f"最大リトライ回数({max_retries})超過: {last_error}")

    response_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    return json.loads(response_text)

def extract_from_pdf(uploaded_file, api_key: str) -> dict:
    """PDFのテキストを抽出してGeminiに送る。画像PDFの場合はページを画像変換して送る"""
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    text = extract_text_from_pdf(io.BytesIO(pdf_bytes))

    if text.strip():
        # テキストPDF：抽出テキストをそのまま送信
        parts = [{"text": EXTRACTION_PROMPT + "\nテキスト:\n" + text[:3000]}]
        return call_gemini_api(api_key, parts), text[:500]
    else:
        # 画像PDF（スキャン等）：1ページ目を画像化してGeminiに送信
        img_bytes = render_pdf_as_image(pdf_bytes)
        image_data = base64.b64encode(img_bytes).decode("utf-8")
        parts = [
            {"text": EXTRACTION_PROMPT},
            {"inline_data": {"mime_type": "image/png", "data": image_data}}
        ]
        return call_gemini_api(api_key, parts), "[画像PDF - 画像として解析]"

def compress_image(uploaded_file, max_size=(1600, 1600), quality=85) -> bytes:
    """画像をリサイズ・圧縮してJPEGバイトで返す"""
    img = Image.open(uploaded_file)
    img = img.convert("RGB")  # PNG等もJPEGに変換
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def extract_from_image(uploaded_file, api_key: str) -> dict:
    """画像を圧縮してbase64エンコードしGeminiに送る"""
    compressed = compress_image(uploaded_file)
    image_data = base64.b64encode(compressed).decode("utf-8")
    parts = [
        {"text": EXTRACTION_PROMPT},
        {"inline_data": {"mime_type": "image/jpeg", "data": image_data}}
    ]
    return call_gemini_api(api_key, parts), "[画像ファイル]"

def build_mf_row(data: dict, default_debit: str, default_credit: str) -> dict:
    amount = int(data.get("税込合計金額", 0))
    tax = int(data.get("消費税額", 0))
    return {
        "取引日": data.get("取引日", datetime.today().strftime("%Y/%m/%d")),
        "決算整理": "",
        "借方勘定科目": data.get("推奨借方勘定科目") or default_debit,
        "借方補助科目": "",
        "借方部門": "",
        "借方税区分": data.get("税区分", "課税仕入10%"),
        "借方金額": amount,
        "借方税額": tax,
        "貸方勘定科目": default_credit,  # 常にデフォルト設定を使用（AIの判断は無視）
        "貸方補助科目": "",
        "貸方部門": "",
        "貸方税区分": "",
        "貸方金額": amount,
        "貸方税額": "",
        "摘要": data.get("摘要", data.get("支払先", "")),
        "仕訳メモ": data.get("備考", ""),
        "タグ": "",
        "MF仕訳ID": "",
    }

# ============================
# SharePoint アップロード
# ============================
def get_sharepoint_token(tenant_id, client_id, client_secret):
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    }
    resp = requests.post(url, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def upload_to_sharepoint(token, site_id, folder_path, filename, content):
    folder_path = folder_path.strip("/")
    url = (
        f"https://graph.microsoft.com/v1.0/sites/{site_id}"
        f"/drive/root:/{folder_path}/{filename}:/content"
    )
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "text/csv"}
    resp = requests.put(url, headers=headers, data=content)
    resp.raise_for_status()
    return resp.json().get("webUrl", "")

# ============================
# Streamlit UI
# ============================
st.set_page_config(
    page_title="領収書 → マネーフォワード 仕訳変換",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 領収書 → マネーフォワード 仕訳CSV 変換ツール")
st.markdown("PDF・画像をアップロード → AIで自動解析 → MFにインポートできるCSVを生成・SharePointに保存")

saved_key = get_gemini_key()

# ============================
# サイドバー
# ============================
with st.sidebar:
    st.header("⚙️ 設定")

    # Gemini API
    st.subheader("🤖 Gemini API")
    if saved_key:
        st.success("✅ APIキー設定済み（入力不要）")
        api_key = saved_key
    else:
        api_key = st.text_input(
            "Gemini APIキー",
            type="password",
            placeholder="AIza...",
            help="https://aistudio.google.com/ で無料取得できます"
        )

    st.markdown("---")

    # SharePoint
    st.subheader("☁️ SharePoint 設定")
    sp_configured = False
    try:
        tenant_id     = st.secrets["SP_TENANT_ID"]
        client_id     = st.secrets["SP_CLIENT_ID"]
        client_secret = st.secrets["SP_CLIENT_SECRET"]
        site_id       = st.secrets["SP_SITE_ID"]
        sp_folder     = st.secrets["SP_FOLDER"]
        sp_configured = True
        st.success("✅ SharePoint設定済み")
    except Exception:
        with st.expander("SharePoint接続情報を入力", expanded=False):
            tenant_id     = st.text_input("テナントID")
            client_id     = st.text_input("クライアントID")
            client_secret = st.text_input("クライアントシークレット", type="password")
            site_id       = st.text_input("サイトID")
            sp_folder     = st.text_input("保存フォルダパス", placeholder="経理/領収書CSV")
        sp_configured = all([tenant_id, client_id, client_secret, site_id, sp_folder])

    st.markdown("---")

    # デフォルト設定
    st.subheader("📋 デフォルト設定")
    st.caption("⬇️ AIが判断できなかった場合のフォールバック")
    default_debit  = st.selectbox("デフォルト借方勘定科目", DEBIT_ACCOUNTS, index=DEBIT_ACCOUNTS.index("消耗品費"))
    st.caption("⬇️ 常にこの科目を使用（AI判断に関わらず固定）")
    credit_options = ["--- 選択してください ---"] + CREDIT_ACCOUNTS
    default_credit_sel = st.selectbox("貸方勘定科目（固定）", credit_options, index=0)
    default_credit = None if default_credit_sel == "--- 選択してください ---" else default_credit_sel
    if default_credit is None:
        st.warning("⚠️ 貸方勘定科目を選択してください")

    st.markdown("---")
    st.markdown("**📌 使い方**")
    st.markdown("""
1. PDF・画像をアップロード（複数可）
2. 「変換開始」をクリック
3. 内容を確認・修正
4. SharePointに保存
5. MFでインポート
""")

# ============================
# メイン：ファイルアップロード
# ============================
uploaded_files = st.file_uploader(
    "📂 領収書をアップロード（PDF・PNG・JPG・JPEG、複数選択可）",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} 件のファイルを読み込みました")

    if not api_key:
        st.warning("⚠️ サイドバーにGemini APIキーを入力してください")
        st.stop()

    if default_credit is None:
        st.button("🚀 変換開始", type="primary", use_container_width=True, disabled=True)
        st.warning("⚠️ サイドバーの「貸方勘定科目（固定）」を選択してから変換してください")
        st.stop()

    if st.button("🚀 変換開始", type="primary", use_container_width=True):
        results = []
        progress = st.progress(0)
        status = st.empty()

        file_bytes_dict = {}
        for i, uploaded_file in enumerate(uploaded_files):
            status.info(f"処理中: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            ext = get_file_ext(uploaded_file.name)

            # ファイルのバイトデータを保存（プレビュー用）
            uploaded_file.seek(0)
            file_bytes_dict[uploaded_file.name] = uploaded_file.read()
            uploaded_file.seek(0)

            try:
                if ext == "pdf":
                    extracted, preview = extract_from_pdf(uploaded_file, api_key)
                else:
                    extracted, preview = extract_from_image(uploaded_file, api_key)

                row = build_mf_row(extracted, default_debit, default_credit)
                row["_ファイル名"] = uploaded_file.name
                row["_生テキスト"] = preview
                results.append(row)
            except Exception as e:
                st.error(f"❌ {uploaded_file.name}: エラー ({e})")

            progress.progress((i + 1) / len(uploaded_files))

        status.success(f"✅ 変換完了！ {len(results)}/{len(uploaded_files)} 件を処理しました")
        if results:
            st.session_state["results"] = results
            st.session_state["file_bytes"] = file_bytes_dict

# ============================
# 結果の表示・編集・出力
# ============================
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    file_bytes = st.session_state.get("file_bytes", {})

    st.markdown("---")
    st.subheader("📋 変換結果（編集可能）")

    # ============================
    # 2カラム：左=編集テーブル / 右=プレビュー
    # ============================
    col_table, col_preview = st.columns([6, 4])

    with col_table:
        st.caption("📝 科目・金額は表内で直接編集できます。プレビュー表示する証憑は右側のラジオボタンで選択してください。")
        display_cols = [c for c in MF_COLUMNS if c in results[0]]

        # 選択中の行インデックスをセッションで管理
        if "selected_row_idx" not in st.session_state:
            st.session_state["selected_row_idx"] = 0
        selected_idx = min(st.session_state["selected_row_idx"], len(results) - 1)

        df_base = pd.DataFrame(results)[["_ファイル名"] + display_cols]

        edited_df = st.data_editor(
            df_base,
            column_config={
                "_ファイル名": st.column_config.TextColumn("ファイル名", disabled=True, width="medium"),
                "取引日": st.column_config.TextColumn("取引日", width="small"),
                "借方勘定科目": st.column_config.SelectboxColumn("借方勘定科目", options=DEBIT_ACCOUNTS),
                "借方税区分": st.column_config.SelectboxColumn("借方税区分", options=TAX_CATEGORIES),
                "貸方勘定科目": st.column_config.SelectboxColumn("貸方勘定科目", options=CREDIT_ACCOUNTS),
                "借方金額": st.column_config.NumberColumn("借方金額（税込）", format="%d"),
                "借方税額": st.column_config.NumberColumn("借方税額", format="%d"),
                "貸方金額": st.column_config.NumberColumn("貸方金額", format="%d"),
            },
            num_rows="dynamic",
            use_container_width=True,
            height=500,
            hide_index=True,
            key="mf_data_editor",
        )

    with col_preview:
        st.markdown("**🔍 元ファイルプレビュー**")

        # ラジオボタンでプレビュー対象の証憑を選択（真のラジオボタン）
        file_options = edited_df["_ファイル名"].tolist() if len(edited_df) > 0 else []

        if file_options:
            # 表示ラベルは「No.行番号: ファイル名」形式にして視認性UP
            label_map = {f"{i+1}. {name}": i for i, name in enumerate(file_options)}
            labels = list(label_map.keys())
            default_label = labels[min(selected_idx, len(labels) - 1)]

            chosen_label = st.radio(
                "プレビューする証憑",
                options=labels,
                index=labels.index(default_label),
                key="preview_radio",
                label_visibility="collapsed",
            )
            new_idx = label_map[chosen_label]
            if new_idx != selected_idx:
                st.session_state["selected_row_idx"] = new_idx
                st.rerun()

            preview_file = file_options[new_idx]
        else:
            preview_file = None

        if preview_file:
            st.caption(f"📄 {preview_file}")

        if preview_file and preview_file in file_bytes:
            ext = get_file_ext(preview_file)
            b = file_bytes[preview_file]

            if ext == "pdf":
                try:
                    img_bytes = render_pdf_as_image(b)
                    st.image(img_bytes, use_column_width=True)
                    page_count = len(fitz.open(stream=b, filetype="pdf"))
                    if page_count > 1:
                        st.caption(f"※ 1ページ目を表示中（全{page_count}ページ）")
                except Exception as e:
                    st.warning(f"PDFプレビューエラー: {e}")
            else:
                st.image(b, use_column_width=True)
        else:
            st.info("ラジオボタンで証憑を選択すると表示されます")

    with st.expander("📄 抽出テキストを確認"):
        for r in results:
            st.markdown(f"**{r['_ファイル名']}**")
            st.text(r.get("_生テキスト", ""))
            st.markdown("---")

    # CSV生成
    export_df = edited_df[[c for c in MF_COLUMNS if c in edited_df.columns]]
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")
    filename = f"mf_仕訳インポート_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv"

    st.markdown("---")
    st.subheader("📤 出力")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**① PCにダウンロード**")
        st.download_button(
            label="⬇️ CSVをダウンロード",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    with col2:
        st.markdown("**② SharePointに保存**")
        if not sp_configured:
            st.info("サイドバーのSharePoint設定を入力してください")
        else:
            if st.button("☁️ SharePointに保存", type="primary", use_container_width=True):
                with st.spinner("SharePointにアップロード中..."):
                    try:
                        token = get_sharepoint_token(tenant_id, client_id, client_secret)
                        web_url = upload_to_sharepoint(
                            token, site_id, sp_folder, filename, csv_bytes
                        )
                        st.success("✅ SharePointに保存しました！")
                        if web_url:
                            st.markdown(f"[📁 ファイルを開く]({web_url})")
                    except Exception as e:
                        st.error(f"❌ エラー: {e}")

    st.markdown("---")
    st.info("MFへのインポート手順：会計帳簿 → 仕訳帳 → インポート → 仕訳帳 → CSVを選択")
    st.metric("変換件数", f"{len(export_df)} 件")
