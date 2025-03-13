import { Chat } from "@/components/Chat/Chat"
import { Navbar } from "@/components/Layout/Navbar"
import { Message } from "@/types"
import Head from "next/head"
import React from "react"
import { useEffect, useRef, useState } from "react"

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [initializing, setInitializing] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const chatbotContent = `Sag mir zuerst, **WAS** ich tun soll (Aktion), dann **WOMIT* (Text), getrennt durch einen Doppelpunkt \nBeispiel: Erstelle mir ein Zitat von: ...`;

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSend = async (message: Message) => {
    const updatedMessages = [...messages, message];
    setMessages(updatedMessages);
    setLoading(true);

    const request: RequestInit = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: { role: "user", content: message.content} as Message} ),
    };

    const response = await fetch("/api/chat", request);
    if (!response.ok) {
      setLoading(false);
      throw new Error(response.statusText);
    }
    const responseData = await response.json();
    const messageContent = responseData.message?.content;

    if (!messageContent) {
      return;
    }

    setLoading(false);
    let done = false;
    let isFirst = true;

    while (!done) {
      done = true;
      if (isFirst) {
        isFirst = false;
        setMessages((messages) => [
          ...messages,
          {
            role: "assistant",
            content: messageContent
          }
        ]);
      } else {
        setMessages((messages) => {
          const lastMessage = messages[messages.length - 1];
          const updatedMessage = {
            ...lastMessage,
            content: lastMessage.content + messageContent
          };
          return [...messages.slice(0, -1), updatedMessage];
        });
      }
    }
  };

  const resetInputState = async () => {
  try {
    const response = await fetch("http://localhost:8080/reset", { method: "POST" });
    if (!response.ok) {
      throw new Error("Reset failed");
    }
    const data = await response.json();
    console.log(data.response);
  } catch (error) {
    console.error("Error resetting input state:", error);
  }
};

  const handleReset = (chatbotContent: string) => {
    resetInputState();
    setMessages([
      {
        role: "assistant",
        content: chatbotContent
      }
    ]);
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
      } catch (error : any) {
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
        <meta
          name="description"
          content=""
        />
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1"
        />
      </Head>
      <div className="flex flex-col h-screen">
        <Navbar />
        <div className="flex-1 flex items-center justify-center overflow-auto sm:px-10 pb-4 sm:pb-10 py-10">
          <div className="max-w-[1200px] w-full mx-auto" style={{ maxHeight: 'calc(100vh - var(--navbar-height) - 20px)' }}>
            {initializing ? (
              errorMessage !== '' ? (
                <>
                  <div className="text-center text-lg font-bold">Initialisierungsfehler:</div>
                  <div className="text-center text-lg">&quot;{errorMessage}&quot;</div>
                </>
              ) : (
                <div className="text-center text-lg font-bold">Initialisierung...</div>
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
