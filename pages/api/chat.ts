import { Message } from "@/types";
import { spawn } from "child_process";
import { NextApiRequest, NextApiResponse } from "next";

const handler = async (req: NextApiRequest, res: NextApiResponse): Promise<NextApiResponse> => {
  const body = req.body;
  const message: Message = body?.message || { role: "user", content: "Test" };
  console.log(message.content);

  const pythonVersion = process.env.PYTHON_EXECUTABLE || "python3";
  const pythonProcess = spawn(pythonVersion, ["./main.py", message.content.toString()]);

  //show pythonprocess output
  pythonProcess.stdout.on("data", (data) => {
    console.log(`stdout: ${data}`);
  });

  let output = "";
  let errorOutput = "";

  // Listen for data output
  pythonProcess.stdout.on("data", (data) => {
    output += data.toString(); // Accumulate standard output
  });

  // Listen for errors
  pythonProcess.stderr.on("data", (data) => {
    errorOutput += data.toString(); // Accumulate error output
  });

  let response_message: Message = { role: "assistant", content: ""}

  // On process close
  pythonProcess.on("close", (code) => {
    if (code === 0) {
      // If successful, send the result back as JSON
      try {
        const jsonMatch = output.match(/\{.*\}/); // Matches the first JSON object in the output
        if (jsonMatch) {
          const parsedContent = JSON.parse(jsonMatch[0]); // Parse the matched JSON string
          response_message.content = parsedContent.message
          res.status(200).json({message: response_message}); // Return only the JSON part
        } else {
          response_message.content = "No valid JSON found in output";
          res.status(200).json({message: response_message});
        }
      } catch (parseError) {
        response_message.content = "Failed to parse JSON";
        res.status(500).json({message: response_message});
      }
    } else {
      response_message.content = "Python script failed: " + errorOutput;
      res.status(500).json({message: response_message});
    }
  });
  return res;
};

export default handler;