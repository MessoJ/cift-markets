DO $$
DECLARE
    v_user_id UUID;
    v_bank_id UUID;
    v_card_id UUID;
    v_wallet_id UUID;
BEGIN
    -- Get the first user
    SELECT id INTO v_user_id FROM users ORDER BY id LIMIT 1;
    
    IF v_user_id IS NULL THEN
        RAISE NOTICE 'No users found.';
        RETURN;
    END IF;

    RAISE NOTICE 'Populating funding data for user ID: %', v_user_id;

    -- 1. Payment Methods
    -- Check if any exist
    IF NOT EXISTS (SELECT 1 FROM payment_methods WHERE user_id = v_user_id) THEN
        
        -- Insert Bank Account
        INSERT INTO payment_methods (
            user_id, type, name, bank_name, account_number_encrypted, routing_number,
            last_four, is_default, is_verified, created_at, updated_at
        ) VALUES (
            v_user_id, 'bank_account', 'Chase Checking', 'Chase Bank', 'encrypted_1234', '123456789',
            '1234', TRUE, TRUE, NOW(), NOW()
        ) RETURNING id INTO v_bank_id;

        -- Insert Debit Card
        INSERT INTO payment_methods (
            user_id, type, name, card_brand, last_four, card_exp_month, card_exp_year,
            is_default, is_verified, created_at, updated_at
        ) VALUES (
            v_user_id, 'debit_card', 'Visa Debit', 'Visa', '4242', 12, 2028,
            FALSE, TRUE, NOW(), NOW()
        ) RETURNING id INTO v_card_id;

        -- Insert Crypto Wallet
        INSERT INTO payment_methods (
            user_id, type, name, crypto_network, crypto_address, last_four,
            is_default, is_verified, created_at, updated_at
        ) VALUES (
            v_user_id, 'crypto_wallet', 'Metamask', 'ethereum', '0x71C...9A23', '9A23',
            FALSE, TRUE, NOW(), NOW()
        ) RETURNING id INTO v_wallet_id;
        
        RAISE NOTICE 'Added payment methods.';
    ELSE
        -- Fetch existing IDs for transaction creation
        SELECT id INTO v_bank_id FROM payment_methods WHERE user_id = v_user_id AND type = 'bank_account' LIMIT 1;
        SELECT id INTO v_card_id FROM payment_methods WHERE user_id = v_user_id AND type = 'debit_card' LIMIT 1;
        RAISE NOTICE 'Payment methods already exist.';
    END IF;

    -- 2. Funding Transactions
    IF NOT EXISTS (SELECT 1 FROM funding_transactions WHERE user_id = v_user_id) THEN
        
        IF v_bank_id IS NOT NULL THEN
            -- Completed Deposit
            INSERT INTO funding_transactions (
                user_id, type, amount, status, payment_method_id, method,
                created_at, completed_at
            ) VALUES (
                v_user_id, 'deposit', 5000.00, 'completed', v_bank_id, 'bank_transfer',
                NOW() - INTERVAL '30 days', NOW() - INTERVAL '28 days'
            );

            -- Completed Withdrawal
            INSERT INTO funding_transactions (
                user_id, type, amount, status, payment_method_id, method,
                created_at, completed_at
            ) VALUES (
                v_user_id, 'withdrawal', 200.00, 'completed', v_bank_id, 'bank_transfer',
                NOW() - INTERVAL '5 days', NOW() - INTERVAL '3 days'
            );
            
            -- Processing Deposit
            INSERT INTO funding_transactions (
                user_id, type, amount, status, payment_method_id, method,
                created_at, completed_at
            ) VALUES (
                v_user_id, 'deposit', 10000.00, 'processing', v_bank_id, 'bank_transfer',
                NOW() - INTERVAL '2 hours', NULL
            );
        END IF;

        IF v_card_id IS NOT NULL THEN
            -- Completed Card Deposit
            INSERT INTO funding_transactions (
                user_id, type, amount, status, payment_method_id, method,
                created_at, completed_at
            ) VALUES (
                v_user_id, 'deposit', 1500.00, 'completed', v_card_id, 'card',
                NOW() - INTERVAL '15 days', NOW() - INTERVAL '15 days'
            );
        END IF;

        RAISE NOTICE 'Added funding history.';
    ELSE
        RAISE NOTICE 'Funding history already exists.';
    END IF;

END $$;
