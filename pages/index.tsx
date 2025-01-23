import { Chat } from "@/components/Chat/Chat";
import { Navbar } from "@/components/Layout/Navbar";
import { Message } from "@/types";
import Head from "next/head";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [initializing, setInitializing] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const chatbotContent = `Wie kann ich dir helfen?`;

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSend = async (message: Message) => {
    setMessages([...messages, message]);
    setLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: { role: "user", content: message.content } as Message,
        }),
      });

      if (!response.ok) {
        throw new Error(response.statusText);
      }

      const responseData = await response.json();
      const messageContent = responseData.generated_text; 

      if (!messageContent) {
        console.error("Flan-T5 hat keine Antwort generiert.");
        setErrorMessage("Flan-T5 hat keine Antwort generiert.");
        return;
      }

      setMessages((messages) => [
        ...messages,
        {
          role: "assistant",
          content: messageContent,
        },
      ]);
    } catch (error: any) {
      console.error("Fehler beim Senden der Nachricht:", error);
      setErrorMessage(`Fehler: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = (chatbotContent: string) => {
    setMessages([
      {
        role: "assistant",
        content: chatbotContent,
      },
    ]);
    setErrorMessage(""); // Fehlermeldung zurücksetzen
  };

  useEffect(() => {
    const initializeChatbot = async () => {
      try {
        const response = await fetch("/api/initialization");
        if (!response.ok) {
          throw new Error(response.statusText);
        }
        setMessages([
          {
            role: "assistant",
            content: chatbotContent,
          },
        ]);
        setInitializing(false);
      } catch (error: any) {
        setErrorMessage(error.toString());
      }
    };
    initializeChatbot();
  }, [chatbotContent]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <>
      <Head>
        <title>LLM-Tutor</title>
        <meta name="description" content="" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <div className="flex flex-col h-screen">
        <Navbar />
        <div className="flex-1 flex items-center justify-center overflow-auto sm:px-10 pb-4 sm:pb-10 py-10">
          <div
            className="max-w-[1200px] w-full mx-auto"
            style={{ maxHeight: "calc(100vh - var(--navbar-height) - 20px)" }}
          >
            {initializing ? (
              errorMessage !== "" ? (
                <>
                  <div className="text-center text-lg font-bold">
                    Initialisierungsfehler:
                  </div>
                  <div className="text-center text-lg">
                    &quot;{errorMessage}&quot;
                  </div>
                </>
              ) : (
                <div className="text-center text-lg font-bold">
                  Initialisierung...
                </div>
              )
            ) : (
              <>
                <Chat
                  messages={messages}
                  loading={loading}
                  onSend={handleSend}
                  onReset={() => handleReset(chatbotContent)}
                />
                <div ref={messagesEndRef} />
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}