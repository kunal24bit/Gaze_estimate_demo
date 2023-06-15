
from typing import Union, Tuple
import cv2
from fastapi import FastAPI, File, Request
from fastapi.param_functions import Form
from iris_tracking import process_frame
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory = "templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/track_iris")
async def track_iris(request: Request, threshold: float = Form(...)) -> Tuple[str, float]:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        iris_pos, ratio = process_frame(frame, threshold)

        cv2.putText(frame,"Iris_pos and ratio are {}, {}".format(iris_pos, ratio),(30, 30),cv2.FONT_HERSHEY_PLAIN ,1.2, (255,0, 0), 1, cv2.LINE_AA)

        cv2.imshow("Iris Tracking", frame)

        if cv2.waitKey(1) ==  ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

    return iris_pos, ratio

import uvicorn
if __name__ =="_main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)

