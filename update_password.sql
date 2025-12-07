-- Update admin password with proper bcrypt hash
UPDATE users 
SET hashed_password = '$2b$12$ENgu8JHNTTn0LJW0G2lUtuE4tBESzzNqcuGbQCV64FmvdXTi8Bi8K' 
WHERE email = 'admin@ciftmarkets.com';

-- Verify the update
SELECT email, hashed_password FROM users WHERE email = 'admin@ciftmarkets.com';
