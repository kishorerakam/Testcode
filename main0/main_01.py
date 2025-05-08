from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import csv

app = FastAPI()

# Allow frontend to call API from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_csv(content: str, expected_delimiter: int) -> dict:
    lines = content.strip().splitlines()
    if not lines:
        return {"status": "fail", "reason": "Empty file"}
    try:
        delimiter = csv.Sniffer().sniff(lines[0]).delimiter
    except:
        return {"status": "fail", "reason": "Could not detect delimiter"}
    
    data_lines = lines[1:]
    counts = [line.count(delimiter) for line in data_lines]
    most_common = max(set(counts), key=counts.count)

    if most_common != expected_delimiter:
        return {"status": "fail", "reason": f"Expected {expected_delimiter}, found {most_common}"}
    return {"status": "success"}

@app.post("/api/validate-csv")
async def validate_csv_only(
    file: UploadFile = File(...),
    expected_delimiter: int = Form(...)
):
    content = await file.read()
    text = content.decode("utf-8")
    result = validate_csv(text, expected_delimiter)
    if result["status"] != "success":
        raise HTTPException(status_code=400, detail=result["reason"])
    return {"status": "success"}
