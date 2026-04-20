import os
import requests

# ✅ Automatically Face_Segmentation_Project folder mein celebrity_faces banayega
FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "celebrity_faces")
os.makedirs(FACES_DIR, exist_ok=True)

celebrities = {
    "Scarlett_Johansson"     : "https://tse1.mm.bing.net/th/id/OIP.Tor5K9KfJ-FvCFLQ_d5nHgHaEo?rs=1&pid=ImgDetMain&o=7&rm=3",
    "Chris_Hemsworth"        : "https://fr.web.img6.acsta.net/pictures/19/06/05/12/04/5240675.jpg",
    "Robert_Downey_Jr"       : "https://static1.srcdn.com/wordpress/wp-content/uploads/2024/07/instar53643496.jpg",
    "Chris_Evans"            : "https://hips.hearstapps.com/hmg-prod/images/chris-evans-gettyimages-1138769185.jpg?resize=1200:*",
    "Tom_Holland"            : "https://wallpapercave.com/wp/wp6938294.jpg",
    "Zendaya"                : "https://image.tmdb.org/t/p/original/6WPolY7Wd3GMiuN1dPxYZX7liik.jpg",
    "Benedict_Cumberbatch"   : "https://tse3.mm.bing.net/th/id/OIP.s_zZ4kLIPt8qpqMUaGZMuAHaLH?rs=1&pid=ImgDetMain&o=7&rm=3",
    "Josh_Brolin"            : "https://tse3.mm.bing.net/th/id/OIP.V2neM50MXukPDiQ25JF6swHaKu?rs=1&pid=ImgDetMain&o=7&rm=3",
    "Dwayne_Johnson"         : "https://thepersonage.com/wp-content/uploads/2020/10/Dwayne-Douglas-Johnson.jpg",
    "Ram_Charan"             : "https://w0.peakpx.com/wallpaper/404/436/HD-wallpaper-ramcharan-actor-movies-ram-charan-ram-charan-tej-rrr-telugu-actor-charan-telugu-movies-bollywood-ram.jpg",
    "N_T_Rama_Rao_Jr"        : "https://bigstarbio.com/wp-content/uploads/2020/04/Jr.-NTR-e1588082316527-1024x916.jpg",
}

print("Downloading celebrity faces...")
downloaded = []
for name, url in celebrities.items():
    try:
        save_path = os.path.join(FACES_DIR, f"{name}.jpg")
        response  = requests.get(url, timeout=15,
                                 headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            downloaded.append(name)
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} — status {response.status_code}")
    except Exception as e:
        print(f"  ❌ {name} — {e}")

print(f"\nDownloaded: {len(downloaded)}/{len(celebrities)}")
print(f"Saved to  : {FACES_DIR}")