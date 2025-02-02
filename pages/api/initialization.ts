import { spawn } from "child_process";
import { NextApiRequest, NextApiResponse } from "next";

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  const pythonVersion = process.env.PYTHON_EXECUTABLE || "python3";
  const pythonProcess = spawn(pythonVersion, ["./main.py", req.body.text]);

  let result = "";
  let error = "";

  pythonProcess.stdout.on("data", (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    error += data.toString();
  });

  pythonProcess.on("close", (code) => {
    if (code === 0) {
      try {
        const parsedResult = JSON.parse(result);
        res.status(200).json(parsedResult);
      } catch (parseError) {
        res.status(500).json({ error: "Failed to parse Python response", details: result.trim() });
      }
    } else {
      res.status(500).json({
        error: "Python script failed",
        details: error.trim() || "Unknown error occurred",
      });
    }
  });

  pythonProcess.on("error", (err) => {
    res.status(500).json({
      error: "Failed to execute Python script",
      details: err.message,
    });
  });
};

export default handler;
