-------------------------------------------------------------------------------
-- Stone Soup Numeric Types Boundary Value Tests Specification
--
-- This file tests the domain-specific numeric types at their boundaries
-- to ensure range constraints and safe operations work correctly.
--
-- Author: Stone Soup Contributors
-- Version: 0.1.0
-------------------------------------------------------------------------------

with AUnit.Test_Suites;
with AUnit.Test_Cases;

package Test_Numeric_Types is

   -- Test case type
   type Numeric_Test_Case is new AUnit.Test_Cases.Test_Case with null record;

   -- Required override for AUnit abstract function
   overriding
   function Name (T : Numeric_Test_Case) return AUnit.Message_String;

   -- Required override for registering test routines
   overriding
   procedure Register_Tests (T : in out Numeric_Test_Case);

   -- Return the test suite
   function Suite return AUnit.Test_Suites.Access_Test_Suite;

end Test_Numeric_Types;
