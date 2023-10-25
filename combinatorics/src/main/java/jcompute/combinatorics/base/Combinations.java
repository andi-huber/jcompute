/*
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */
package jcompute.combinatorics.base;

import java.math.BigInteger;

import lombok.experimental.UtilityClass;

@UtilityClass
public class Combinations {

    /**
     * Also known as {@literal n choose k}.
     * @return {@code n! / (k!(n-k)!)}
     */
    public BigInteger binomial(final int n, final int k) {
        if ((n == k) || (k == 0)) {
            return BigInteger.ONE;
        }
        if (n < k) {
            return BigInteger.ZERO;
        }
        if (k > (n >> 1)) {
            return binomial(n, n - k);
        }
        BigInteger nominator = BigInteger.ONE;
        BigInteger divisor = BigInteger.ONE;
        for (int i = 1; i <= k; i++) {
            divisor = divisor.multiply(BigInteger.valueOf(i));
            nominator = nominator.multiply(BigInteger.valueOf(n - i + 1));
        }
        return nominator.divide(divisor);
    }

    /**
     * Also known as {@literal n choose k}.
     * @return {@code n! / (k!(n-k)!)}
     */
    public long binomialAsLongValueExact(final int n, final int k) {
        if ((n == k) || (k == 0)) {
            return 1;
        }
        if (n < k) {
            return 0;
        }
        return binomial(n, k).longValueExact();
    }

    public long next_colex(long x) {
        long r = x & -x; // lowest set bit
        x += r; // replace lowest block by a one left to it
        //      if ( 0==x ) return 0; // input was last combination
        long z = x & -x; // first zero beyond lowest block
        z -= r; // lowest block (cf. lowest_block())

        //while ( 0==(z&1) ) { z >>= 1; } // move block to low end of word
        //return x | (z>>1); // need one bit less of low block

        // better on sparse systems ...
        return x | (z>>Long.numberOfTrailingZeros(z)+1);
    }

}
