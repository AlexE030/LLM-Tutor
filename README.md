# LLM-Tutor

Dieser Code enthält Inhalte aus dem Rechtchecker Repository: https://github.com/AnnaKohlbecker/RechtChecker

Dieses ist unter der MIT-Lizenz lizensiert.

Das Projekt ist ausgelegt um auf dem KI-Server der DHBW zu laufen. Die Lauffähigkeit auf anderen Geräten kann nicht garantiert werden.

Im Folgenden wird das aufsetzen den Projekts erklärt:

## 1. Holen des Projektordners
Entweder:
```
git clone https://github.com/AlexE030/LLM-Tutor.git
```
Oder:
Entpacken des Mitgelieferten zip-Ordners im Zielverzeichnis

## 2. Erstellen von virtual env

```
python -m venv venv
```

## 3. Virtuelle Umgebung aktivieren

cmd:
```
venv\Scripts\activate
```

powershell:
```
.\venv\Scripts\Activate.ps1
```
bash:
```
source venv/bin/activate
```

## 4. Installieren der Frontend Dependencies
```
npm install 
```

## 5. Nötige Einstellungen bei Huggingface
1. Melden Sie sich mit Ihrem Huggingface-Konto auf https://huggingface.co an.
2. Akzeptieren Sie die Lizenzbedingungen unter https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
3. Erstellen Sie einen API-Key unter Settings -> Access Tokens. Dieser sollte mindestens Read-Rechte haben.

## 6. Definieren der .env
Erstellen sie eine Datei .env auf Basis der .env.example.
Geben sie hierfür an:

1. Den Pfad zu Ihrer Python executable in virtualenv
2. Ihren Huggingface API-Key

## 7. Erstellen der docker-container
```
docker-compose up -d --build
```
Dieser Vorgang kann einige Zeit in Anspruch nehmen.
Nachdem der Build abgeschlossen ist, sollte nochmal etwa 15 Minuten gewartet werden, damit alle LLM richtig starten können.

## Start
```
npm run dev
```

## Known Issues 
- Manchmal kann es passieren, das trotz erstmaliger Eingabe direkt nach der Spezifizierung der Anfrage gefragt wird. Das lässt sich recht leicht beheben, indem man auf den Button Neuer Chat klickt.
- Manchmal wird zurückgegeben, dass kein Kontext gegeben ist. In diesem Fall auch den Chat neu laden und die Anfrage nochmal stellen.