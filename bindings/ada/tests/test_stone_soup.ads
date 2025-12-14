-------------------------------------------------------------------------------
-- Stone Soup Ada Unit Tests Specification
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

with AUnit.Test_Suites;
with AUnit.Test_Cases;

package Test_Stone_Soup is

   -- Test case type
   type Test_Case is new AUnit.Test_Cases.Test_Case with null record;

   -- Required override for AUnit abstract function
   overriding
   function Name (T : Test_Case) return AUnit.Message_String;

   -- Required override for registering test routines
   overriding
   procedure Register_Tests (T : in out Test_Case);

   -- Return the test suite
   function Suite return AUnit.Test_Suites.Access_Test_Suite;

end Test_Stone_Soup;
