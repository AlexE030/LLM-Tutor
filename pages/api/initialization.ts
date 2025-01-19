import { spawn } from "child_process";
import { NextApiRequest, NextApiResponse } from "next";

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  const pythonVersion = process.env.PYTHON_EXECUTABLE || "python3";
  const pythonProcess = spawn(pythonVersion, ["./main.py"]);

  let result = "";
  let error = "";

  // Listen for standard output
  pythonProcess.stdout.on("data", (data) => {
    result += data.toString(); // Accumulate standard output
  });

  // Listen for error output
  pythonProcess.stderr.on("data", (data) => {
    error += data.toString(); // Accumulate error output
  });

  // Handle process close
  pythonProcess.on("close", (code) => {
    if (code === 0) {
      // Successfully executed
      try {
        const parsedResult = JSON.parse(result); // Attempt to parse JSON
        res.status(200).json(parsedResult);
      } catch (parseError) {
        // Send raw result if JSON parsing fails
        res.status(200).json({ output: result.trim() });
      }
    } else {
      // Error occurred
      res.status(500).json({
        error: "Python script failed",
        details: error.trim() || "Unknown error occurred",
      });
    }
  });

  // Add error handling for spawn failure
  pythonProcess.on("error", (err) => {
    res.status(500).json({
      error: "Failed to execute Python script",
      details: err.message,
    });
  });
};

export default handler