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
package jcompute.combinatorics.finspace;

import java.math.BigInteger;
import java.util.Optional;
import java.util.function.IntFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import jcompute.core.util.function.MultiIntConsumer;
import jcompute.core.util.function.MultiIntPredicate;
import jcompute.core.util.function.PrefixedMultiIntConsumer;

///Given n dimensions, the finite space is spanned by given dimension sizes s1 x s2 x .. x sn.
///
///Example:
/// ```
///  // print all 3 x 2 tuples
///  var fs = FiniteSpace.create(3, 2);
///  fs.forEachSequential(v->System.out.println("tuple: %d, %d".formatted(v[0], v[1])));
/// ```
///Caveats / Future Work:
///- no `Stream<int[]>` creation
///- no branch pruning support
///- no computation cancellation support
///- no progress indication callback support
public interface FiniteSpace {

    enum Visiting {
        SEQUENTIAL{
            @Override
            IntStream range(final int upperExclusive) {
                return IntStream.range(0, upperExclusive);
            }
        },
        PARALLEL{
            @Override
            IntStream range(final int upperExclusive) {
                return IntStream.range(0, upperExclusive).parallel();
            }
        };
        abstract IntStream range(int upperExclusive);
    }

    /**
     * Number of distinct vectors in this space.
     */
    BigInteger cardinality();
    /**
     * Number of dimensions this space has.
     */
    int dimensionCount();
    /**
     * Reports the dimension sizes to the consumer. (single call)
     */
    void reportDimensionSizes(MultiIntConsumer intConsumer);

    /**
     * Visits all distinct vectors.
     */
    void forEach(Visiting visiting, MultiIntConsumer intConsumer);
    default void forEachSequential(final MultiIntConsumer intConsumer) {
        forEach(Visiting.SEQUENTIAL, intConsumer);
    }
    default void forEachParallel(final MultiIntConsumer intConsumer) {
        forEach(Visiting.PARALLEL, intConsumer);
    }

    /**
     * Creates a collector for each possible integer of the first dimension, then streams them after they passed given prefixedIntConsumer.
     */
    <T> Stream<T> streamCollectors(final IntFunction<T> collectorFactory, final PrefixedMultiIntConsumer<T> prefixedIntConsumer);


    /**
     * Visits up to all distinct vectors an optionally returns any that matches given predicate.
     */
    Optional<int[]> findAny(MultiIntPredicate intPredicate);

    // -- Factory

    public static FiniteSpace create(final int... dim) {
        if(dim==null
            || dim.length==0
            || IntStream.of(dim).anyMatch(size->size<=0)) return new FiniteSpace0();

        return switch (dim.length) {
            case 1 -> new FiniteSpace1(dim[0]);
            case 2 -> new FiniteSpace2(dim[0], dim[1]);
            case 3 -> new FiniteSpace3(dim[0], dim[1], dim[2]);
            case 4 -> new FiniteSpace4(dim[0], dim[1], dim[2], dim[3]);
            case 5 -> new FiniteSpace5(dim[0], dim[1], dim[2], dim[3], dim[4]);
            case 6 -> new FiniteSpace6(dim[0], dim[1], dim[2], dim[3], dim[4],
                dim[5]);
            case 7 -> new FiniteSpace7(dim[0], dim[1], dim[2], dim[3], dim[4],
                dim[5], dim[6]);
            case 8 -> new FiniteSpace8(dim[0], dim[1], dim[2], dim[3], dim[4],
                dim[5], dim[6], dim[7]);
//            case 8 -> new FiniteSpaceComposite(
//                new FiniteSpace3(dim[0], dim[1], dim[2]),
//                new FiniteSpace5(dim[3], dim[4], dim[5], dim[6], dim[7]));
            default -> new FiniteSpaceN(dim);
        };
    }

}
