import React from 'react';
import Navbar from './Navbar';
import Sidebar from './Sidebar';
import { useAppContext } from '../../contexts/AppContext';
import { Error } from '../common/Error';
import { Loading } from '../common/Loading';

const MainLayout = ({ children }) => {
  const { state } = useAppContext();
  const { loading, error } = state;

  return (
    <div className="flex h-screen flex-col">
      <Navbar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-auto p-4 bg-gray-50">
          {loading && <Loading />}
          {error && <Error message={error} />}
          {!loading && !error && children}
        </main>
      </div>
    </div>
  );
};

export default MainLayout;