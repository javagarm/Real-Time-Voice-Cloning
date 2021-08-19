
from fastapi import FastAPI, UploadFile, File
import os

app = FastAPI()

@app.post("/create_file/")
async def image(audio: UploadFile = File(...)):
    #print(audio.filename)
    # print('../'+os.path.isdir(os.getcwd()+"images"),"*************")
    try:
        os.mkdir("audios")
        #print(os.getcwd())
    except Exception as e:
        print("suee") 
    file_name = os.getcwd()+"/audios/"+audio.filename.replace(" ", "-").replace("\"", "").replace("\'", "")
    with open(file_name,'wb+') as f:
        f.write(audio.file.read())
        f.close()
    in_fpath = "audios/"+audio.filename
    print(in_fpath)    
    return {"success"}