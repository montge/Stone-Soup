-------------------------------------------------------------------------------
-- Stone Soup Unit Scaling Test Main Program
--
-- Standalone test program for unit scaling
--
-- Build with: gnatmake -P stonesoup_tests.gpr test_unit_scaling_main
-- Run with: ./obj/test_unit_scaling_main
-------------------------------------------------------------------------------

with Test_Unit_Scaling;

procedure Test_Unit_Scaling_Main is
begin
   Test_Unit_Scaling.Run_All_Tests;
end Test_Unit_Scaling_Main;
