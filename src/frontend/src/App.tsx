import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { CaseStatusProvider } from "./context/CaseStatusContext";
import { AccountDetail } from "./pages/AccountDetail";
import { Dashboard } from "./pages/Dashboard";

function App() {
  return (
    <CaseStatusProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/accounts/:accountId" element={<AccountDetail />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </CaseStatusProvider>
  );
}

export default App;
