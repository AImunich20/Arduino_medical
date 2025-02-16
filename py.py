import { useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FaUser, FaLock } from "react-icons/fa";

export default function AuthPage() {
  const [form, setForm] = useState({ email: "", password: "" });

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white">
      <motion.div initial={{ opacity: 0, y: -50 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <Card className="w-96 shadow-2xl bg-gray-800 rounded-2xl">
          <CardContent className="p-6">
            <Tabs defaultValue="login" className="w-full">
              <TabsList className="flex justify-center bg-gray-700 p-1 rounded-full">
                <TabsTrigger value="login" className="flex-1">Login</TabsTrigger>
                <TabsTrigger value="signup" className="flex-1">Sign Up</TabsTrigger>
              </TabsList>
              
              <TabsContent value="login">
                <AuthForm form={form} setForm={setForm} type="login" />
              </TabsContent>
              <TabsContent value="signup">
                <AuthForm form={form} setForm={setForm} type="signup" />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

function AuthForm({ form, setForm, type }) {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
      <div className="space-y-4">
        <div className="relative">
          <FaUser className="absolute left-3 top-3 text-gray-400" />
          <Input
            type="email"
            placeholder="Email"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
            className="pl-10 bg-gray-700 text-white border-none"
          />
        </div>
        <div className="relative">
          <FaLock className="absolute left-3 top-3 text-gray-400" />
          <Input
            type="password"
            placeholder="Password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            className="pl-10 bg-gray-700 text-white border-none"
          />
        </div>
        <Button className="w-full bg-blue-500 hover:bg-blue-600">
          {type === "login" ? "Login" : "Sign Up"}
        </Button>
      </div>
    </motion.div>
  );
}
