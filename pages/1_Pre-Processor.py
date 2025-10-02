import io, zipfile, openpyxl, streamlit as st, tempfile

NEW_HEADINGS = ["Cycle_Number","Time","Date","Voltage (mV)","Current (mA)","Capacity (mAh)","Energy (mWh)"]
COLUMNS_TO_DELETE = [3, 9, 10, 11, 12, 13]

def _preproc(b: bytes, name: str):
    issues = []
    try:
        wb = openpyxl.load_workbook(io.BytesIO(b))
    except Exception as e:
        return None, f"modified_{name}", [f"{name}: load error: {e}"]
    names = wb.sheetnames
    if len(names) < 2:
        return None, f"modified_{name}", [f"{name}: not enough sheets; skipped."]
    ws = wb[names[1]]
    for sn in list(names):
        if sn != names[1]:
            wb.remove(wb[sn])
    for col in sorted(COLUMNS_TO_DELETE, reverse=True):
        try:
            ws.delete_cols(col)
        except Exception as e:
            issues.append(f"{name}: delete col {col} failed: {e}")
    for i, h in enumerate(NEW_HEADINGS, 1):
        ws.cell(row=1, column=i, value=h)
    base = name.rsplit(".", 1)[0]
    try:
        ws.title = base[:31]
    except Exception:
        ws.title = "Sheet1"
    out = io.BytesIO(); wb.save(out); out.seek(0)
    return out.read(), f"modified_{name}", issues

st.set_page_config(page_title="Pre-Processor", page_icon="🛠️", layout="wide")
st.title("🛠️ Pre-Processor")

uploaded = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)
auto_continue = st.checkbox("Send output to Raw Processing automatically", value=True)

if st.button("Run Pre-Processor") and uploaded:
    modified, report = [], []
    for f in uploaded:
        try:
            b = f.read(); f.seek(0)
            out_b, out_n, issues = _preproc(b, f.name)
            report.extend(issues or [])
            if out_b is not None:
                modified.append((out_n, out_b))
        except Exception as e:
            report.append(f"{f.name}: unexpected error: {e}")

    if not modified:
        st.warning("No files produced.")
    else:
        with st.expander("Download (optional)", expanded=False):
            if len(modified) == 1:
                nm, data = modified[0]
                st.download_button(f"Download {nm}", data, nm,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for nm, data in modified:
                        zf.writestr(nm, data)
                buf.seek(0)
                st.download_button("Download all (ZIP)", buf, "modified_excels.zip", "application/zip")

        if auto_continue:
            class TempUpload:
                def __init__(self, data: bytes, name: str):
                    self._fh = tempfile.SpooledTemporaryFile(max_size=10_000_000, mode="w+b", suffix=".xlsx")
                    self._fh.write(data); self._fh.seek(0)
                    self.name = name
                def read(self, *a, **k): return self._fh.read(*a, **k)
                def seek(self, *a, **k): return self._fh.seek(*a, **k)
                def tell(self, *a, **k): return self._fh.tell(*a, **k)
                def readable(self): return True
                def seekable(self): return True
                def close(self): return self._fh.close()
            st.session_state["_forwarded_preproc_files"] = [TempUpload(b, n) for n, b in modified]
            st.success("Forwarded. Open the main page and choose Raw cycler Excel.")

if 'uploaded' not in locals():
    st.info("If no uploader appears, confirm the file path: pages/1_Pre-Processor.py and restart Streamlit.")
