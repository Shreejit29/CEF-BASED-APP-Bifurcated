# preprocessor.py
import io
import zipfile
import openpyxl
import streamlit as st

COLUMNS_TO_DELETE = [3, 9, 10, 11, 12, 13]  # 1-based for openpyxl
NEW_HEADINGS = [
    "Cycle number",
    "Time",
    "Date",
    "Voltage (mV)",
    "Current (mA)",
    "Capacity (mAh)",
    "Energy (mWh)"
]

def _process_workbook_bytes(file_bytes: bytes, filename: str):
    issues = []
    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes))
    except Exception as e:
        issues.append(f"{filename}: load error: {e}")
        return None, f"modified_{filename}", issues

    sheet_names = wb.sheetnames
    if len(sheet_names) < 2:
        issues.append(f"{filename}: not enough sheets; skipped.")
        return None, f"modified_{filename}", issues

    second_sheet_name = sheet_names[1]
    ws = wb[second_sheet_name]

    # remove all other sheets
    for sn in list(sheet_names):
        if sn != second_sheet_name:
            wb.remove(wb[sn])

    # delete specified columns (descending)
    for col in sorted(COLUMNS_TO_DELETE, reverse=True):
        try:
            ws.delete_cols(col)
        except Exception as e:
            issues.append(f"{filename}: delete col {col} failed: {e}")

    # set new header row
    for idx, heading in enumerate(NEW_HEADINGS, start=1):
        ws.cell(row=1, column=idx, value=heading)

    # rename sheet to base filename (without extension)
    base = filename.rsplit(".", 1)[0]
    ws.title = base

    # serialize workbook to bytes
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return out.read(), f"modified_{filename}", issues

def render_preprocessor():
    st.subheader("Pre-Processor")
    st.caption("Upload .xlsx files; keeps only the 2nd sheet, deletes columns [3,9,10,11,12,13], rewrites headers, and renames sheet to filename.")
    uploaded = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

    if st.button("Run Pre-Processor") and uploaded:
        modified = []
        report = []
        for f in uploaded:
            try:
                content = f.read()
                f.seek(0)
                out_bytes, out_name, issues = _process_workbook_bytes(content, f.name)
                if issues:
                    report.extend(issues)
                if out_bytes is not None:
                    modified.append((out_name, out_bytes))
            except Exception as e:
                report.append(f"{f.name}: unexpected error: {e}")

        if modified:
            if len(modified) == 1:
                name, data = modified[0]
                st.download_button(
                    f"Download {name}",
                    data=data,
                    file_name=name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            else:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for name, data in modified:
                        zf.writestr(name, data)
                buf.seek(0)
                st.download_button(
                    "Download all (ZIP)",
                    data=buf,
                    file_name="modified_excels.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            st.success("Pre-processing complete.")
        else:
            st.warning("No files produced. Check the report below.")

        if report:
            st.markdown("#### Report")
            for line in report:
                st.write("-", line)
