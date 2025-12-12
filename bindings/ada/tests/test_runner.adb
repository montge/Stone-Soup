-------------------------------------------------------------------------------
-- Stone Soup Ada Test Runner
--
-- Run with: gnatmake -P stonesoup_tests.gpr && ./obj/test_runner
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

with AUnit.Reporter.Text;
with AUnit.Run;
with Test_Stone_Soup;

procedure Test_Runner is
   procedure Run is new AUnit.Run.Test_Runner (Test_Stone_Soup.Suite);
   Reporter : AUnit.Reporter.Text.Text_Reporter;
begin
   Run (Reporter);
end Test_Runner;
