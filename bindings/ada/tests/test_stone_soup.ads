-------------------------------------------------------------------------------
-- Stone Soup Ada Unit Tests Specification
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

with AUnit.Test_Suites;
with AUnit.Test_Fixtures;

package Test_Stone_Soup is

   -- Test fixture type (for use with Test_Caller)
   type Test_Case is new AUnit.Test_Fixtures.Test_Fixture with null record;

   -- Set_Up is called before each test
   overriding
   procedure Set_Up (T : in out Test_Case);

   -- Return the test suite
   function Suite return AUnit.Test_Suites.Access_Test_Suite;

end Test_Stone_Soup;
