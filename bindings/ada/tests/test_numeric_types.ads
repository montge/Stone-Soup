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

   -- Return the test suite
   function Suite return AUnit.Test_Suites.Access_Test_Suite;

end Test_Numeric_Types;
