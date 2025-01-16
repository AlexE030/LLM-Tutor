import React, { FC } from "react";
import { ScaleIcon } from "@heroicons/react/24/solid"; // Heroicon for "Law/Justice"

export const Navbar: FC = () => {
  return (
    <div className="relative flex flex-col bg-white py-4 items-center justify-between">
      <div className="flex items-center space-x-2 font-bold text-3xl">
        <p>LLM-Tutor</p>
      </div>
      <div className="w-full mt-2">
        <div className="h-[4px] bg-black"></div>
      </div>
    </div>
  );
};
