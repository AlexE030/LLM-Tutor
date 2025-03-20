import { spawn } from "child_process";
import { NextApiRequest, NextApiResponse } from "next";

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  const pythonExecutable = process.env.PYTHON_EXECUTABLE || "python3";

  // Zuerst das Initialisierungsskript ausführen
  const initProcess = spawn(pythonExecutable, ["./init_data.py"]);
  let initOutput = "";
  let initError = "";

  initProcess.stdout.on("data", (data) => {
    initOutput += data.toString();
  });

  initProcess.stderr.on("data", (data) => {
    initError += data.toString();
  });

  initProcess.on("close", (initCode) => {
    if (initCode === 0) {
      console.log("Initialisierung erfolgreich:", initOutput);
      // Nachdem init_data.py erfolgreich abgeschlossen wurde, starte main.py
      const mainProcess = spawn(pythonExecutable, ["./main.py", req.body.text]);
      let mainOutput = "";
      let mainError = "";

      mainProcess.stdout.on("data", (data) => {
        mainOutput += data.toString();
      });

      mainProcess.stderr.on("data", (data) => {
        mainError += data.toString();
      });

      mainProcess.on("close", (mainCode) => {
        if (mainCode === 0) {
          try {
            const parsed = JSON.parse(mainOutput);
            res.status(200).json(parsed);
          } catch (parseError) {
            res.status(200).json({ output: mainOutput.trim() });
          }
        } else {
          res.status(500).json({
            error: "main.py failed",
            details: mainError.trim() || "Unknown error in main.py",
          });
        }
      });

      mainProcess.on("error", (err) => {
        res.status(500).json({
          error: "Failed to execute main.py",
          details: err.message,
        });
      });
    } else {
      res.status(500).json({
        error: "init_data.py failed",
        details: initError.trim() || "Unknown error in init_data.py",
      });
    }
  });

  initProcess.on("error", (err) => {
    res.status(500).json({
      error: "Failed to execute init_data.py",
      details: err.message,
    });
  });
};

export default handler;
